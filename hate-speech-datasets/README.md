# Perspective-API Scores for the Multilingual Hate-Speech Supersets

This bundle contains every post in the ten **hate-speech supersets** released
with Tonneau et al. (WOAH 2024), enriched with **Perspective API** scores
computed in May 2026.

## Contents

One Parquet file per language, named `<language>.parquet`. Row count:

| File                  | Rows    |
|-----------------------|--------:|
| `arabic.parquet`      | 449,078 |
| `english.parquet`     | 360,493 |
| `german.parquet`      |  58,024 |
| `kenya.parquet`       |  48,076 |
| `portuguese.parquet`  |  43,222 |
| `turkish.parquet`     |  41,423 |
| `spanish.parquet`     |  29,855 |
| `french.parquet`      |  18,071 |
| `indonesian.parquet`  |  14,306 |
| `india.parquet`       |  14,155 |
| **Total**             | **1,076,703** |

## Columns

Original superset columns:

- `text` — the annotated post
- `labels` — binary annotation (1 = hateful, 0 = not hateful)
- `source` — platform / medium (e.g. `Twitter`, `Reddit`, `NGO`, ...)
- `dataset` — upstream dataset identifier (see "Upstream datasets" below)
- `nb_annotators` — number of annotators on this post (where available)
- `post_author_country_location` — inferred country of the post's author
- `_split` — the HF split this row originated from (most are `train`-only)
- `_row_idx` — row index within the loaded superset (preserves original order)

Perspective API scores (probability in [0, 1]):

- `pscore_TOXICITY`
- `pscore_SEVERE_TOXICITY`
- `pscore_IDENTITY_ATTACK`
- `pscore_INSULT`
- `pscore_PROFANITY`
- `pscore_THREAT`

English only (experimental Perspective attributes, requested for `english.parquet`):

- `pscore_AFFINITY_EXPERIMENTAL`
- `pscore_COMPASSION_EXPERIMENTAL`
- `pscore_CURIOSITY_EXPERIMENTAL`
- `pscore_NUANCE_EXPERIMENTAL`
- `pscore_PERSONAL_STORY_EXPERIMENTAL`
- `pscore_REASONING_EXPERIMENTAL`
- `pscore_RESPECT_EXPERIMENTAL`

Error column:

- `pscore__error` — non-null when the Perspective request failed for that row.
  Across all 1,076,703 rows there are **26 errors total** (mostly empty strings
  or comments exceeding the Perspective 20kB limit).

## How to load

```python
import pandas as pd
df = pd.read_parquet("english.parquet")
df.head()
```

or with `pyarrow` / `polars` / `duckdb` — any standard Parquet reader works.

## How the scores were produced

Each row's `text` was sent to the Perspective API
(`commentanalyzer.googleapis.com/v1alpha1/comments:analyze`) with
`doNotStore: true`. The Perspective `languages` hint used per superset:

| Superset    | Language hint |
|-------------|---------------|
| arabic      | `ar`          |
| english     | `en`          |
| french      | `fr`          |
| german      | `de`          |
| indonesian  | `id`          |
| portuguese  | `pt`          |
| spanish     | `es`          |
| turkish     | `en` *        |
| india       | `en` *        |
| kenya       | `en` *        |

\* Turkish is not modeled by Perspective; the India and Kenya supersets
contain code-switched Hindi/English and Swahili/English. For these three we
forced the English model rather than letting requests fail. Scores for these
files should therefore be interpreted with extra caution.

Texts longer than 20,000 bytes (Perspective's hard limit) were truncated
before scoring.

## Upstream datasets

The supersets aggregate previously-published hate-speech datasets identified
via a systematic survey (early 2024). Per language:

- **Arabic** (10): `L-HSAB`, `T-HSAB`, `OSACT`, `MLMA`, `Let-Mi`,
  `alsafari`, `saudi`, `brothers`, `jhsc`, `aracovid`
- **English** (25): `CONAN`, `ETHOS`, `GHC`, `MLMA`, `hatemoji-build`,
  `call_me_sexist`, `davidson`, `east_asian`, `learning from the worst`,
  `hasoc`, `hateful_symbols`, `melsherief`, `compliment_sexist`,
  `white_supremacy`, `online-misogyny`, `Parler`, `toraman`, `CAD`,
  `measuring-hate-speech`, `EDOS`, `hatexplain`, `anatomy_online_hate`,
  `are_you_a_racist`, `benchmark`, `fox_news`
- **French** (5): `CONAN`, `cyberado`, `MLMA`, `sexism`, `FTR`
- **German** (6): `Bretschneider`, `RP-mod-crowd`, `gahd`, `hasoc`,
  `refugees`, (one unnamed)
- **India** (2): `hasoc`, `hostility detection`
- **Indonesian** (3): `ID_instagram`, `ID_multilabel`, `IDHSD`
- **Kenya** (1): `code-switched`
- **Portuguese** (4): `ToLD-Br`, `TuPy`, `HateBR`, `Fortuna`
- **Spanish** (6): `chileno`, `misocorpus`, `haternet`, `homomex`,
  `hateval`, `hascosva`
- **Turkish** (3): `mayda_et_al_1`, `mayda_et_al_2`, `HATC`

The `dataset` column identifies the upstream source for each row. For the
full upstream citations, see the dataset card on Hugging Face:
`https://huggingface.co/datasets/manueltonneau/<language>-hate-speech-superset`.

## Citation

If you use these scores, please cite the superset paper:

```bibtex
@inproceedings{tonneau-etal-2024-languages,
    title     = "From Languages to Geographies: Towards Evaluating Cultural Bias in Hate Speech Datasets",
    author    = {Tonneau, Manuel and Liu, Diyi and Fraiberger, Samuel and
                 Schroeder, Ralph and Hale, Scott and R{\"o}ttger, Paul},
    booktitle = "Proceedings of the 8th Workshop on Online Abuse and Harms (WOAH 2024)",
    month     = jun,
    year      = "2024",
    address   = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2024.woah-1.23",
    pages     = "283--311"
}
```

and acknowledge the Perspective API
(`https://perspectiveapi.com/`, Jigsaw / Google).

## Access & ethics

The upstream supersets on Hugging Face are **gated**: users must accept terms
before download, including a commitment **not to use the data to cause harm
to human subjects** and **not to train generative LLMs to produce hateful
content**. By accepting this bundle you agree to the same conditions.

Intended use: training/evaluating hate-speech detection models and studying
hateful discourse online.

Posts contain offensive and potentially distressing language by design.
