# Perspective API Score Archive for Toxicity & Hate-Speech Benchmarks

A dated snapshot of **Perspective API** scores for **16 publicly available
toxicity and hate-speech datasets**, computed in **May 2026**. It is a record of
the model's final state before Perspective API closes at the end of 2026.

## Contents

One Parquet file per dataset under `benchmarks/` and 'hate-speech-datasets'
(rows, ok/failed counts and languages per dataset). Datasets covered:

- **LLM-evaluation benchmark:** RealToxicityPrompts
- **Annotated detection benchmarks:** HateCheck, HateXplain,
  Measuring Hate Speech, Civil Comments
- **Multilingual hate-speech corpora:** the ten Tonneau et al. (WOAH 2024)
  hate-speech supersets; HateDay

Scores are keyed by a text hash (`uid`); source texts are **not**
redistributed — see *Reconstruction*.

## Columns

- `uid`: sha1 of the whitespace-normalised text (join key)
- `toxicity`, `severe_toxicity`, `identity_attack`, `insult`, `profanity`,
  `threat`:  Perspective probabilities in [0, 1]
- `lang`:  language hint sent with the request
- `query_ts` : UTC request timestamp (the de-facto model version pin)
- `ok`, `error` : request status; failed rows have null scores
- `detected_languages`, `http_status` : status and so forth

## Reconstruction

To attach scores to text: download the original dataset, normalise each text
(collapse whitespace), take the first 16 hex chars of its sha1, and join on
`uid`.

## How scores were produced

Each text was sent to `commentanalyzer.googleapis.com/v1alpha1/comments:analyze`
with `doNotStore: true`, requesting the six production attributes. Texts above
Perspective's 20 kB limit were skipped.

## Citation

If you use this archive, please cite [THIS PAPER (Bye Bye Perspective API) — anonymised for review] and
acknowledge the Perspective API (Jigsaw / Google, https://perspectiveapi.com/).
Each source dataset keeps its own license and citation, see its dataset card.

## Ethics

Source texts contain offensive and distressing language by design. Intended for
reproducibility, evaluating hate-speech detection, and studying measurement
infrastructure and especially not for training models to generate hateful content.
