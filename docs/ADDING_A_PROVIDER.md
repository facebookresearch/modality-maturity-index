# Adding a system under test

This harness measures whether a system returns the output modalities a prompt
asks for. Adding your own system means writing one adapter. This document is
the contract; `tests/test_provider_conformance.py` is the executable version of
it, and `mmi/providers/stub_provider.py` is a working reference designed to be
copied.

## The contract in one sentence

**Extract, do not adjudicate.**

Your adapter's job is to call your system and hand back what came out. Deciding
what counts as an Image is not your adapter's job, and the harness will not let
it be.

## Why the split is drawn there

The two halves of the problem have opposite shapes.

*How* a system hands you an artifact is unbounded and keeps changing: inline
base64, a file part, a `file_uri`, an `image_generation_call.result`, a
tool-produced file, a signed URL, something that does not exist yet. Nobody can
enumerate that in advance, so it stays in adapters where it can grow.

*What counts* as an artifact is fixed: five modalities bound to MIME families,
set by the benchmark. If each adapter answered "does `image/svg+xml` count as
Image?" or "is a bytes-less URL native?" for itself, then adding a system would
silently redefine the benchmark and two systems' scores would stop being
comparable. So that question is answered once, in `mmi/detection.py`.

When your system ships something the shared table cannot classify, the fix is
one line in that table — strictly better than every adapter carrying its own
opinion.

## What you implement

Subclass `BaseProvider` and implement `_send_impl`:

```python
async def _send_impl(self, prompt: EvalPrompt) -> ProviderResponse: ...
```

Return a `ProviderResponse` with these fields:

| Field | Meaning |
|---|---|
| `response_text` | **User-visible prose only.** Never tool traces. |
| `output_assets` | `list[CapturedAsset]` — every artifact, tagged with how it arrived |
| `tool_calls` | `list[ToolCallRecord]` — observational; never scored |
| `raw_response` | Plain JSON. Never `str()`. Use `self.json_raw_response(...)` |
| `request` | A `RequestRecord`; it is redacted before persistence |

`BaseProvider.finalize()` assembles this for you and runs URL capture over the
prose. Use it.

## Tagging artifacts

Every artifact carries two independent facts.

**`delivery` — how it arrived.** This is structural, and it is what decides
whether the artifact can earn native credit:

| Value | Use it when |
|---|---|
| `PROVIDER_INLINE` | Bytes came back in the model's own response body |
| `PROVIDER_TOOL` | A tool on your platform produced it (code interpreter, image-gen call) |
| `HARNESS_TOOL` | One of MMI's neutral media tools produced it |
| `EXTERNAL_URL` | It only ever appeared as a URL |

`PROVIDER_INLINE` and `PROVIDER_TOOL` are recorded distinctly so that "the model
emitted it" can later be separated from "the model's platform tool emitted it",
without re-running anything.

**`capture_status` — whether we hold the bytes.** `capture_bytes()` and
`capture_reference()` set this for you. `captured` requires actual bytes; a
reference you could not resolve is `reference_only`.

The scoring class is *derived* from those two and is deliberately not something
you can set:

| delivery | capture_status | scores as |
|---|---|---|
| inline / platform tool / harness tool | `captured` | **native** (strict + lenient) |
| inline / platform tool / harness tool | anything else | nothing |
| external URL | any | **URL class** (lenient only) |

## Rules

**Do not decide modalities.** Pass the MIME type through and let
`mmi.detection.classify_mime` decide. If your MIME is genuinely opaque — for
example `application/octet-stream` from an endpoint that is authoritative about
what it produced — you may set `modality_hint`. It is recorded in the trace, and
the shared table wins wherever it can classify, so a hint can never mint credit
the table would have refused.

**Do not put tool traces in `response_text`.** Only what a user would actually
see. This is the rule that stops a URL inside a hidden tool call from earning
URL credit, and it holds regardless of how capable your system is. It gets more
load-bearing as models gain tools, not less.

**Do not require bytes you do not have.** If your system reports success without
returning an artifact, record a `capture_reference(...)`. A claim is not a
production, and the metric needs to tell them apart.

**Do not fetch remote content yourself.** `mmi/fetch.py` does that, with SSRF
protections. Accept a `fetch_remote_assets: bool = False` keyword argument and
forward it to `super().__init__(...)`; the harness sets it when rubric scoring is
on, because a rubric grades an artifact's content and there is nothing to grade
without the bytes. Pass it to `capture_url(..., enabled=self.fetch_remote_assets)`
rather than calling any fetcher directly. Retrieval never changes provenance:
`delivery` stays `external_url`, so a fetched URL cannot become native credit.

**Do not classify by hostname.** There is no CDN allowlist. Provenance is
structural.

**Do not score.** No `detected_native`, no `pass_strict`. The conformance suite
fails your adapter if those tokens appear in your module.

## Registering your provider

1. Add an entry to `PROVIDER_REGISTRY` in `mmi/config.py` with the environment
   variable holding your API key and an empty `base_url` (meaning "your SDK's
   own official endpoint"). A non-empty `base_url` is opt-in custom routing and
   must never cause the harness to inject vendor-specific headers.
2. Add a `case` to `_make_provider` in `mmi/runner.py`. Every provider is
   constructed with the same `common` kwargs, so your `__init__` must accept
   `fetch_remote_assets` or construction fails.
3. Export it from `mmi/providers/__init__.py`.
4. Add your adapter to `PROVIDER_CASES` in
   `tests/test_provider_conformance.py`.

## Capabilities

If your system has built-in search, browsing, or its own tools, that is fine.
Nothing rejects it. Express provider-native tools with a verbatim
`provider_tools` block in your TOML:

```toml
[[models]]
name = "my-system"
provider = "mysystem"
model_id = "my-model-v1"
provider_tools = [
  { type = "web_search" },
]
```

The harness passes these through untouched and does not interpret them.

One consequence to understand: **results are comparable only within an identical
capability configuration.** A system with a search tool and a system without one
are not measuring the same thing, and the harness does not pretend otherwise.

## Running the conformance suite

```bash
uv run pytest tests/test_provider_conformance.py -v
```

It checks:

- an artifact with bytes scores native
- a URL in prose scores URL-class, not native
- a tool-produced file records tool provenance
- opaque MIME takes the hint path, the hint is recorded, the shared table is
  consulted first
- a bytes-less reference earns no native credit
- a hidden tool-trace URL earns no URL credit
- a provider error still produces a scored row that counts in the denominator
- `raw_response` is JSON-serializable, never a repr string
- asset IDs are unique
- your module contains no modality-classification logic

If a scenario genuinely cannot be expressed by your system, return `None` from
your factory for it and the suite will skip it.
