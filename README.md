# Modality Maturity Index (MMI)

The Modality Maturity Index (MMI) is a benchmark designed to evaluate the multimodal capabilities of LLMs across five modalities and combinations of up to three modalities in both inputs and outputs.
This repository contains the harness for obtaining MMI scores.
The MMI data (893 prompts, five modalities, per-modality human-written rubrics) does not live in this repository, it can be found on Hugging Face: <https://huggingface.co/datasets/facebook/mmi>.
It is loaded from Hugging Face at runtime into the standard HF cache; input media is materialized into a cache directory outside the source tree.
More information about the benchmark can be found in our paper:

> ### [Modality Maturity Index: A benchmark for assessing multimodal capabilities of omni models][paper]
>
> Rohit Patel, Dieuwke Hupkes, Sloan Strader - *Meta Superintelligence Labs*, 2026
>

[paper]: https://arxiv.org/abs/2608.26317

<details>
<summary><b>📚 Cite this work (BibTeX)</b></summary>

```bibtex
@misc{patel2026modalitymaturityindexbenchmark,
  title         = {Modality Maturity Index: A benchmark for assessing
                   multimodal capabilities of omni models},
  author        = {Rohit Patel and Dieuwke Hupkes and Sloan Strader},
  year          = {2026},
  eprint        = {2608.26317},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.26317}
}
```

</details>

## What the harness does

Given one or more models to test - each declared in a TOML config as a provider, a model ID and optionally a custom endpoint - the harness sends every MMI prompt with its input media attached, records what the model says back, analyzes which output modalities the response contains and extracts the assets, and scores those against the prompt's golden set.
It then returns the benchmark's two main metrics as well as several complementary metrics:

| Metric | What it is | Where it is computed |
|---|---|---|
| **Modality Presence Score (MPS)** | Per-prompt F1 between the modalities returned and the gold set, averaged over prompts. Presence only - content is not inspected. | `mmi/scorer.py`, always on |
| **MMI Value** | Rubric score: for each expected output modality, an LLM judge grades each human-written criterion in isolation; criteria are averaged within a modality, then modality means are averaged. | `mmi/rubric_scorer.py`, `rubric_enabled = true` |

Alongside MPS the harness reports precision, recall, and **pass rate** (binary: 1 only if every expected modality is present).
Undesired `Text` is not penalized in precision.
A prompt whose input the provider rejects is scored zero, which is what the paper reports as the input failure rate.

Prompts are sent to models verbatim, with no system prompt and no output-modality steering, because the point is to observe what a system does by default.

## Install

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).
The harness runs from a git checkout; it is not installed as a package.

```bash
git clone https://github.com/facebookresearch/modality-maturity-index
cd modality-maturity-index
uv sync
```

Optional dependency groups:

```bash
uv sync --group dev      # pytest + ruff
uv sync --group viewer   # streamlit result viewer
uv sync --group dataset  # pandas/pyarrow, for local .parquet datasets
```

## Quickstart

Set the API keys for the providers you want to run:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```

Models whose key is not set are skipped with a warning.
Then:

```bash
uv run python -m mmi --config default.toml
```

`configs/default.toml` runs a curated **25-prompt sample** against three example models, with the judge and rubric scoring off;
Results land in `results/default/`.

## Configuration

Configs are TOML files in `configs/`, passed by bare filename or by path.
Three config files are included in this repository:

- `configs/default.toml` - 25-prompt sample, three example models, no judge, no tools.
- `configs/paper_main_eval.toml` - the paper's main evaluation: all 893 prompts, judge on, no tools. (Llama 4 Maverick is absent; it was self-served in the paper and no public route for it ships here.)
- `configs/paper_rubric_eval.toml` - the paper's rubric-validation experiment: three driver models with neutral image/audio/video generation tools, rubric scoring on.

### `[settings]`

| Key | Default | Meaning |
|---|---|---|
| `sample` | `true` | Restrict to the 25 curated prompt IDs in `mmi/dataset.py`. Set `false` for all 893. |
| `prompt_ids` | `[]` | Run exactly these prompt IDs. Takes precedence over `sample`. Unknown IDs are an error. |
| `max_prompts` | unset | Truncate to the first N prompts (only when `sample` is off). |
| `concurrency` | `3` | Concurrent in-flight prompts per model. |
| `max_retries` / `retry_backoff` | `5` / `4` | Retry policy for transient provider failures. Permanent failures (404/401/…) are never retried. |
| `request_timeout` | `300` | Per-call timeout, seconds. The rubric config raises this to 900 for slow video generation. |
| `runner_timeout_padding` | `60` | Added to `request_timeout` for the outer runner deadline. |
| `verbose` | `false` | DEBUG-level logging. |
| `judge_enabled` | `false` | Enable the structural fallback judge (Layer 2 detection). |
| `judge_provider` / `judge_model` | `google` / `gemini-3-flash-preview` | Judge routing. `google` is the only preset. |
| `rubric_enabled` | `false` | Grade rubrics inline during the run, producing the MMI Value. Also turns on retrieval of URL-delivered artifacts, since a rubric needs bytes to grade. |
| `rubric_judge_model` | falls back to `judge_model` | Rubric judge routing. |
| `tool_loop_limit` | `3` | Maximum rounds of neutral media-tool calls per prompt. |

### `[[models]]`

```toml
[[models]]
name = "gpt-5-example"      # run name; also the results filename
provider = "openai"         # openai | anthropic | gemini | stub
model_id = "gpt-5.4"        # defaults to `name`
api = "responses"           # OpenAI only: responses | chat
tools = ["image_gen"]       # neutral MMI media tools to offer
provider_tools = [{ type = "web_search" }]  # verbatim provider-native tools, passed through untouched
base_url = ""               # empty = the SDK's own official endpoint
api_key_env = ""            # override the provider default key variable
```

All models run with a 32,768-token generation limit.
Provider-level `response_modalities` are deliberately not forwarded: they would steer the answer's modality independently of the prompt.

Results are comparable only within an identical capability configuration.
A system with a search tool and one without are not measuring the same thing, and the harness does not pretend otherwise.

### `[media_tools]`

Three neutral tools - `image_gen`, `audio_gen`, `video_gen` - each taking a single free-text `prompt` argument and carrying byte-identical, deliberately uninformative descriptions, so the only signal about a tool is its name.
Their schemas are frozen in `mmi/media_tools.py`; only which backend fulfils them is configuration:

```toml
[media_tools.image_gen]
provider = "google"                  # only `google` is implemented
model = "gemini-3.1-flash-image"
```

A tool with no configured backend is simply not offered to the model.
The resolved backend is recorded in the run manifest.

## CLI

```
uv run python -m mmi [--config NAME] [--resume]
                     [--rejudge | --rescore PATH | --rubric-score PATH]
                     [--dataset-path PATH]
```

| Mode | What it does |
|---|---|
| *(default)* | Run every model in the config over the selected prompts, score, and write results + summary. |
| `--resume` | Continue the most recent run in the config's results directory. Refuses to resume if the run manifest is incompatible (different dataset revision, prompt set, or config hash). |
| `--rejudge` | Run the structural judge over existing results without re-calling any model. Writes `*_rejudged.jsonl`. Forces the judge on regardless of config. |
| `--rescore PATH` | Recompute precision/recall/F1/pass flags from the detection booleans already stored in a JSONL. No API calls at all. Writes `*_rescored.jsonl`. |
| `--rubric-score PATH` | Run the rubric judge over an existing JSONL, one criterion at a time. Writes `*_rubric_scored.jsonl`; resumable. |
| `--dataset-path PATH` | Load a local `.jsonl`/`.parquet` dataset instead of the Hub (same as `MMI_DATASET_PATH`). |

Each mode is idempotent and re-entrant: results are appended per prompt as they complete, so an interrupted run picks up where it stopped.

## Detection: how a modality is counted as present

Three routes for modality detection are implemented (all are in `mmi/detection.py`):

1. **Native** - the system returned actual bytes of the right MIME family, delivered inline, by a platform tool, or by one of MMI's neutral media tools.
2. **URL** - the response prose contains a link to an asset of that modality, matched against platform patterns (YouTube, Flickr, SoundCloud, …) and file extensions.
3. **Judge** - if neither fired, an LLM judge reads the raw response and says whether the modality was in fact produced. A fallback for parser false negatives only.

That yields two standards:

- **Strict**: native only.
- **Lenient**: any of the three. This is the default reported everywhere, on the grounds that a working link to the right video answers the request whether or not the model rendered the pixels.

Provenance is **structural, not hostname-based** - there is no CDN allowlist.
Retrieving a URL's bytes (which rubric scoring does) never promotes it to native credit; `delivery` stays `external_url`.

## Outputs

```
results/<config-name>/
  run_manifest.json                          # dataset revision, prompt-set hash, config hash, model routes
  <timestamp>_<model>.jsonl                  # one scored row per prompt
  <timestamp>_<model>_assets/<prompt_id>/…   # captured artifacts
  <timestamp>_summary.json                   # per-model, per-modality aggregates
```

Each JSONL row carries the per-modality detection booleans, strict/lenient pass flags, P/R/F1, the redacted request trace, the raw provider response, captured assets, tool-call records and - when enabled - judge reasoning and rubric grades.
Traces are redacted through an **allowlist** (`mmi/redaction.py`) before being written, so authorization headers, signed-URL credentials and absolute machine paths do not end up in a results file.

Rubric summaries report three things side by side: the mean MMI Value, the paper's binary per-modality metric (1 only if *every* criterion for that modality passed), and the count of zeros that came from the judge or from artifact retrieval failing rather than from the model.

## Result viewer

A Streamlit app for browsing runs prompt by prompt, with input attachments and returned media rendered inline:

```bash
uv sync --group viewer
uv run --group viewer streamlit run viewer.py
```

Remote URLs are not fetched on open; set `MMI_VIEWER_ALLOW_REMOTE=1` to opt in.

## Adding a model to test

The harness implements adapters for OpenAI, Anthropic and Gemini.
To evaluate anything else - another vendor's API, a self-hosted model, a whole agent - you should write an adapter class that does two things: call your system with the prompt, and return the user-visible text plus any artifact bytes the response carries.

The following three files cover the details:

- `mmi/providers/stub_provider.py` - a minimal working adapter, meant to be copied as a starting point.
- `docs/ADDING_A_PROVIDER.md` - the full contract: which fields to return, and how to register your provider.
- `tests/test_provider_conformance.py` - add your adapter to `PROVIDER_CASES` and it runs the same checks used for the shipped ones.

```bash
uv run pytest tests/test_provider_conformance.py -v
```

## Environment variables

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` | Provider credentials. `GOOGLE_API_KEY` also drives the judges and the media-tool backends. |
| `MMI_DATASET_PATH` | Load a local `.jsonl`/`.parquet` dataset instead of `facebook/mmi`. |
| `MMI_DATASET_REVISION` | Pin a dataset revision (default: `main`). Recorded in the run manifest. |
| `MMI_INPUT_FILES_DIR` | Where input media is materialized (default: `$XDG_CACHE_HOME/mmi/input_files`). Never the repo. |
| `MMI_RESULTS_DIR` | Where run outputs are written (default: `./results`). |
| `MMI_VIEWER_ALLOW_REMOTE` | `1` lets the viewer render remote URLs inline. |

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

## Reproducing the paper

```bash
uv run python -m mmi --config paper_main_eval.toml     # MPS for the frontier models
uv run python -m mmi --config paper_rubric_eval.toml   # tool-scaffolded rubric-validation run
```

Two caveats.
The model IDs in these configs are the ones the paper used and will stop resolving as vendors retire versions; the files record what was run rather than promise it stays runnable, and the run manifest records what actually answered.
The rubric-validation run exists to produce enough gradeable assets to test the rubrics - it is not a model comparison, and its numbers are not comparable with the main evaluation.

## License and citation

Code and dataset are released under [CC BY 4.0](LICENSE.md).
If you use the benchmark or the harness, please cite the paper; see [CITATION.cff](CITATION.cff).

Contributions are welcome - see [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
