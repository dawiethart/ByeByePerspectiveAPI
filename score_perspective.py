
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("Missing dependency 'pandas'. Run: pip install -r requirements.txt")

try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False

ENDPOINT = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

# Six production attributes (available across the 18 production languages).
PRODUCTION_ATTRS = ["TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK",
                    "INSULT", "PROFANITY", "THREAT"]

# The 18 languages with production-attribute support (verified 2026-05).
# NB: Turkish (tr) and Swahili (sw) are NOT here -> superset rows in those
PRODUCTION_LANGS = {"ar", "zh", "cs", "nl", "en", "fr", "de", "hi", "hi-Latn",
                    "id", "it", "ja", "ko", "pl", "pt", "ru", "es", "sv"}

PERSPECTIVE_MAX_BYTES = 20_000
REQUEST_TIMEOUT = 30
RETRYABLE_HTTP = {429, 500, 502, 503, 504}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# rate limiting -- async token bucket
# --------------------------------------------------------------------------
class TokenBucket:
    """Caps aggregate request rate. acquire() returns once a token is free.

    Starts cold (0 tokens) and allows only a small burst, so the very first
    moment of a run cannot overshoot a hard requests/second quota."""

    def __init__(self, rate: float, burst: float | None = None):
        self.rate = float(rate)
        # bounded burst: 10% of the rate, at least 1
        self.capacity = float(burst) if burst else max(1.0, self.rate * 0.10)
        self.tokens = 0.0                       # start cold -> no opening spike
        self.timestamp: float | None = None

    async def acquire(self) -> None:
        loop = asyncio.get_event_loop()
        if self.timestamp is None:
            self.timestamp = loop.time()
        while True:
            now = loop.time()
            self.tokens = min(self.capacity,
                              self.tokens + (now - self.timestamp) * self.rate)
            self.timestamp = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            await asyncio.sleep((1.0 - self.tokens) / self.rate)


def backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter, capped at 60s."""
    return min(60.0, 1.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.5)


# --------------------------------------------------------------------------
# request / response -- pure, testable
# --------------------------------------------------------------------------
def build_request(text: str, lang: str, attrs: list[str]) -> dict:
    body = {
        "comment": {"text": text},
        "requestedAttributes": {a: {} for a in attrs},
        "doNotStore": True,
    }
    if lang in PRODUCTION_LANGS:
        body["languages"] = [lang]
    return body


def parse_summary(data: dict, attrs: list[str]) -> tuple[dict, str]:
    """Pull summary scores. Returns ({attr_lower: value|None}, detected_langs)."""
    scores = {a.lower(): None for a in PRODUCTION_ATTRS}
    attr_scores = data.get("attributeScores", {})
    for a in attrs:
        val = attr_scores.get(a, {}).get("summaryScore", {}).get("value")
        scores[a.lower()] = val
    detected = data.get("detectedLanguages") or data.get("languages") or []
    return scores, ",".join(map(str, detected))


def is_lang_attr_400(body_text: str) -> bool:
    """Heuristic: a 400 that is about language/attribute support, not bad input."""
    t = body_text.lower()
    return any(s in t for s in ("language", "does not support",
                                "not supported", "attribute"))


# --------------------------------------------------------------------------
# scoring one row
# --------------------------------------------------------------------------
async def score_one(session, bucket: TokenBucket, url: str,
                     uid: str, text: str, lang: str,
                     attrs: list[str], max_retries: int) -> dict:
    result = {"uid": uid, "lang": lang, "query_ts": utc_now(), "ok": False,
              "http_status": None, "requested_attributes": "",
              "detected_languages": "", "error": ""}
    for a in PRODUCTION_ATTRS:
        result[a.lower()] = None

    cur_attrs = list(attrs)
    attempt = 0
    while True:
        attempt += 1
        await bucket.acquire()
        try:
            async with session.post(
                url, json=build_request(text, lang, cur_attrs),
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as resp:
                result["http_status"] = resp.status
                if resp.status == 200:
                    data = await resp.json()
                    scores, detected = parse_summary(data, cur_attrs)
                    result.update(scores)
                    result["detected_languages"] = detected
                    result["requested_attributes"] = ",".join(cur_attrs)
                    result["ok"] = True
                    result["query_ts"] = utc_now()
                    return result

                body_text = await resp.text()
                # unsupported language/attribute -> narrow once to TOXICITY
                if (resp.status == 400 and cur_attrs != ["TOXICITY"]
                        and is_lang_attr_400(body_text)):
                    cur_attrs = ["TOXICITY"]
                    continue
                if resp.status in RETRYABLE_HTTP and attempt <= max_retries:
                    await asyncio.sleep(backoff_seconds(attempt))
                    continue
                result["error"] = f"HTTP {resp.status}: {body_text[:200]}"
                return result

        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            if attempt <= max_retries:
                await asyncio.sleep(backoff_seconds(attempt))
                continue
            result["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            return result


# --------------------------------------------------------------------------
# checkpoint / IO
# --------------------------------------------------------------------------
def load_done(scores_jsonl: Path) -> set[str]:
    """uids already present in the scores file (any status counts as done)."""
    done: set[str] = set()
    if not scores_jsonl.exists():
        return done
    with scores_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["uid"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# --------------------------------------------------------------------------
# async run
# --------------------------------------------------------------------------
async def run(rows: list[dict], url: str, scores_jsonl: Path,
              rate: float, concurrency: int, max_retries: int) -> dict:
    in_q: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)
    out_q: asyncio.Queue = asyncio.Queue(maxsize=10_000)
    bucket = TokenBucket(rate)
    stats = {"ok": 0, "failed": 0, "done": 0}
    total = len(rows)
    start = time.time()

    async def producer():
        for row in rows:
            await in_q.put(row)
        for _ in range(concurrency):
            await in_q.put(None)

    async def worker(session):
        while True:
            row = await in_q.get()
            if row is None:
                await out_q.put(None)
                return
            res = await score_one(session, bucket, url, row["uid"],
                                  row["text"], row["lang"],
                                  PRODUCTION_ATTRS, max_retries)
            await out_q.put(res)

    async def writer():
        finished = 0
        with scores_jsonl.open("a", encoding="utf-8") as fh:
            while finished < concurrency:
                item = await out_q.get()
                if item is None:
                    finished += 1
                    continue
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                stats["done"] += 1
                stats["ok" if item["ok"] else "failed"] += 1
                if stats["done"] % 5_000 == 0:
                    fh.flush()
                    el = time.time() - start
                    rps = stats["done"] / el if el else 0
                    print(f"  {stats['done']:,}/{total:,}  "
                          f"ok={stats['ok']:,} fail={stats['failed']:,}  "
                          f"{rps:.0f} req/s", flush=True)

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(
            producer(),
            writer(),
            *[worker(session) for _ in range(concurrency)],
        )
    stats["elapsed_s"] = round(time.time() - start, 1)
    return stats


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("data/pooled.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/scores"))
    ap.add_argument("--rate", type=float, default=600.0,
                    help="requests/second cap (keep under your granted RPC/s)")
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N unscored rows (smoke test)")
    args = ap.parse_args()

    if not _AIOHTTP_AVAILABLE:
        sys.exit("Missing dependency 'aiohttp'. Run: pip install -r requirements.txt")
    api_key = os.environ.get("PERSPECTIVE_API_KEY")
    if not api_key:
        sys.exit("Set PERSPECTIVE_API_KEY in your environment (do not hard-code it).")
    if not args.input.exists():
        sys.exit(f"  ! {args.input} not found — run Step 3 (dedup.py) first.")

    print("== Step 4: score with Perspective API ==")
    df = pd.read_parquet(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores_jsonl = args.out_dir / "scores.jsonl"

    done = load_done(scores_jsonl)
    todo, skipped = [], 0
    for r in df.itertuples(index=False):
        uid, text, lang = r.uid, r.text, getattr(r, "lang", "und")
        if uid in done:
            continue
        text = "" if text is None else str(text)
        # skip empties / oversize now so they are recorded, not retried forever
        if not text.strip():
            _append(scores_jsonl, uid, lang, "skipped_empty")
            skipped += 1
            continue
        if len(text.encode("utf-8")) > PERSPECTIVE_MAX_BYTES:
            _append(scores_jsonl, uid, lang, "skipped_oversize")
            skipped += 1
            continue
        todo.append({"uid": uid, "text": text, "lang": lang})

    print(f"  input rows      : {len(df):,}")
    print(f"  already scored  : {len(done):,}")
    print(f"  skipped (empty/oversize): {skipped:,}")
    if args.limit:
        todo = todo[:args.limit]
    print(f"  to score now    : {len(todo):,}")
    if not todo:
        print("  nothing to do.")
        return _consolidate(scores_jsonl, args.out_dir)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started = utc_now()
    url = f"{ENDPOINT}?key={api_key}"           # key stays in URL, never logged
    stats = asyncio.run(run(todo, url, scores_jsonl, args.rate,
                            args.concurrency, args.max_retries))

    manifest = {
        "run_id": run_id, "started": started, "finished": utc_now(),
        "endpoint": ENDPOINT, "requested_attributes": PRODUCTION_ATTRS,
        "rate_limit_req_s": args.rate, "concurrency": args.concurrency,
        "n_input_rows": len(df), "n_scored_this_run": stats["done"],
        "n_ok": stats["ok"], "n_failed": stats["failed"],
        "elapsed_s": stats["elapsed_s"],
        "note": ("Perspective exposes no model version; 'query_ts' per row is "
                 "the version pin. Re-runs are append-only and resumable."),
    }
    (args.out_dir / f"run_manifest_{run_id}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n  run {run_id}: {stats['ok']:,} ok, {stats['failed']:,} failed, "
          f"{stats['elapsed_s']}s")
    return _consolidate(scores_jsonl, args.out_dir)


def _append(scores_jsonl: Path, uid: str, lang: str, reason: str) -> None:
    """Record a skipped row so it is not reprocessed on resume."""
    row = {"uid": uid, "lang": lang, "query_ts": utc_now(), "ok": False,
           "http_status": None, "requested_attributes": "",
           "detected_languages": "", "error": reason}
    for a in PRODUCTION_ATTRS:
        row[a.lower()] = None
    with scores_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _consolidate(scores_jsonl: Path, out_dir: Path) -> int:
    """Fold the append-only JSONL into a single deduped scores.parquet."""
    if not scores_jsonl.exists():
        return 0
    rows = [json.loads(ln) for ln in scores_jsonl.read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    if not rows:
        return 0
    out = out_dir.parent / "scores.parquet"
    pd.DataFrame(rows).drop_duplicates("uid", keep="last").to_parquet(
        out, index=False)
    print(f"  consolidated {len(rows):,} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
