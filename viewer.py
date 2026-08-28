# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""MMI Result Viewer — browse evaluation results in a Streamlit app.

This source-checkout utility is not included in wheel or sdist artifacts. From
this repository's root, install and launch it with::

    uv sync --group viewer
    uv run --group viewer streamlit run viewer.py
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
from pathlib import Path

import streamlit as st

from mmi.config import INPUT_FILES_DIR, RESULTS_DIR
from mmi.dataset import load_dataset as harness_load_dataset
from mmi.fetch import redact_url
from mmi.input_files import InvalidInputFilePath, resolve_input_file
from mmi.url_modality_detector import (
    AUDIO_EXTENSIONS,
    classify_url,
    extract_urls,
)

# Shared loaders and paths, so the viewer cannot drift from the harness.
RESULTS_ROOT = RESULTS_DIR

#: Render remote URLs inline. Off by default: a results file contains URLs a
#: model produced, and fetching them would make the viewer issue arbitrary
#: outbound requests on open. Set MMI_VIEWER_ALLOW_REMOTE=1 to opt in.
ALLOW_REMOTE_MEDIA = os.environ.get("MMI_VIEWER_ALLOW_REMOTE") == "1"

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".tiff"}
_AUDIO_SUFFIXES = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"}
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
_DOC_SUFFIXES = {".pdf", ".docx", ".doc", ".csv", ".xlsx", ".xls"}

MODALITY_ICONS = {
    "Text": "📝",
    "Image": "🖼️",
    "Audio": "🔊",
    "Video": "🎬",
    "Document": "📄",
}


# ── data loading (cached) ────────────────────────────────────────────


@st.cache_data
def load_prompts() -> dict[str, dict]:
    """Load the pinned eval dataset keyed by prompt_id, via the shared loader."""
    prompts: dict[str, dict] = {}
    for prompt in harness_load_dataset(require_media=False):
        obj = {
            "prompt_id": prompt.prompt_id,
            "prompt_text": prompt.prompt_text,
            "input_modalities": ", ".join(prompt.input_modalities),
            "output_modalities": ", ".join(prompt.output_modalities),
            "input_files": "\n".join(prompt.input_files),
        }
        obj["_input_modalities"] = list(prompt.input_modalities)
        obj["_output_modalities"] = list(prompt.output_modalities)
        obj["_input_files"] = list(prompt.input_files)
        prompts[obj["prompt_id"]] = obj
    return prompts


@st.cache_data
def discover_configs() -> list[str]:
    """Return list of config subdirectories under results/."""
    return sorted(
        d.name for d in RESULTS_ROOT.iterdir() if d.is_dir() and any(d.glob("*.jsonl"))
    )


@st.cache_data
def discover_runs(config_name: str) -> dict[str, str]:
    """Return {display_label: path_str} for every result JSONL under a config dir."""
    config_dir = RESULTS_ROOT / config_name
    runs: dict[str, str] = {}
    for p in sorted(config_dir.glob("*.jsonl")):
        parts = p.stem.split("_", 2)
        if len(parts) == 3:
            ts, tm, model = parts
            label = f"{model}  ({ts}_{tm})"
        else:
            label = p.stem
        runs[label] = str(p)
    return runs


@st.cache_data
def load_results(path_str: str) -> list[dict]:
    path = Path(path_str)
    results: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


@st.cache_data
def load_summary(config_name: str, run_name: str) -> dict | None:
    """Try to load the summary JSON that corresponds to a run."""
    config_dir = RESULTS_ROOT / config_name
    for p in config_dir.glob("*_summary.json"):
        with open(p) as f:
            data = json.loads(f.read())
        if run_name in data:
            return data[run_name]
    return None


# ── helpers ───────────────────────────────────────────────────────────


def _parse_csv(val: str) -> list[str]:
    return [m.strip() for m in val.split(",") if m.strip()] if val else []


def modality_badge(name: str) -> str:
    icon = MODALITY_ICONS.get(name, "❓")
    return f"{icon} {name}"


def pass_fail(ok: bool) -> str:
    return "✅" if ok else "❌"


def format_rate(k: str, v: dict | float | int) -> str:
    """Format a modality pass rate from summary dict.

    Handles both current format (``{"lenient": 0.95, "strict": 0.80, ...}``)
    and legacy format (plain float, e.g. ``0.95``).
    """
    if isinstance(v, dict):
        lenient = v.get("lenient", 0)
        strict = v.get("strict", 0)
        return f"{modality_badge(k)} **{lenient:.0%}** (strict: {strict:.0%})"
    return f"{modality_badge(k)} **{v:.0%}**"


def extract_model_name(label: str) -> str:
    """Extract the plain model name from a run label like 'gpt-4o  (20260325_230724)'."""
    return label.split("  (")[0] if "  (" in label else label


def get_prompt(prompts: dict, pid: str) -> dict:
    """Look up a prompt, returning a safe empty-ish dict if missing."""
    return prompts.get(
        pid,
        {
            "prompt_text": "",
            "_input_modalities": [],
            "_output_modalities": [],
            "_input_files": [],
        },
    )


def render_input_attachments(input_files: list[str]) -> None:
    """Render input file attachments inline with appropriate widgets."""
    st.markdown("**Attachments:** " + "  ·  ".join(f"`{f}`" for f in input_files))

    images: list[tuple[str, Path]] = []
    for fname in input_files:
        try:
            fpath = resolve_input_file(fname, root=INPUT_FILES_DIR)
        except (InvalidInputFilePath, FileNotFoundError):
            st.caption(f"⚠️ {fname} — invalid or missing input file")
            continue
        suffix = fpath.suffix.lower()

        if suffix in _IMAGE_SUFFIXES:
            images.append((fname, fpath))
            continue

        if suffix in _AUDIO_SUFFIXES:
            if fpath.exists():
                st.audio(str(fpath), format=mimetypes.guess_type(fname)[0])
                st.caption(f"🔊 {fname}")
            else:
                st.caption(f"⚠️ {fname} — file not found")
            continue

        if suffix in _VIDEO_SUFFIXES:
            if fpath.exists():
                st.video(str(fpath), format=mimetypes.guess_type(fname)[0])
                st.caption(f"🎬 {fname}")
            else:
                st.caption(f"⚠️ {fname} — file not found")
            continue

        if suffix == ".pdf":
            if fpath.exists():
                pdf_bytes = fpath.read_bytes()
                # Rendered as a download rather than an inline data: iframe.
                # Injecting model-adjacent bytes into the page as raw HTML is
                # not worth the preview.
                st.download_button(
                    f"📥 Download {fname}",
                    pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                )
            else:
                st.caption(f"⚠️ {fname} — file not found")
            continue

        if suffix in _DOC_SUFFIXES:
            if fpath.exists():
                st.download_button(
                    f"📥 Download {fname}",
                    fpath.read_bytes(),
                    file_name=fname,
                    mime=mimetypes.guess_type(fname)[0] or "application/octet-stream",
                )
            else:
                st.caption(f"⚠️ {fname} — file not found")
            continue

    if images:
        cols = st.columns(min(len(images), 3))
        for idx, (fname, fpath) in enumerate(images):
            with cols[idx % 3]:
                if fpath.exists():
                    st.image(str(fpath), caption=fname)
                else:
                    st.caption(f"⚠️ {fname} — file not found")


def render_response_media(raw_text: str) -> None:
    """List media URLs found in the response.

    URL-class evidence is shown as links, not as embedded media, unless the
    operator opts in. These URLs came from a model, and rendering them inline
    would make opening a results file issue outbound requests to whatever the
    model happened to emit.
    """
    classified: list[tuple[str, str]] = []
    for url in extract_urls(raw_text):
        mod = classify_url(url)
        if mod:
            classified.append((redact_url(url), mod))

    if not classified:
        return

    st.markdown("**Media referenced by URL in the response** (URL-class evidence):")
    for url, mod in classified:
        icon = MODALITY_ICONS.get(mod, "🔗")
        if not ALLOW_REMOTE_MEDIA:
            st.markdown(f"{icon} `{mod}` — {url}")
            continue
        if mod == "Image":
            st.image(url)
        elif mod == "Video":
            st.video(url)
        elif mod == "Audio" and re.search(AUDIO_EXTENSIONS, url.lower()):
            st.audio(url)
        else:
            st.markdown(f"{icon} [{url}]({url})")
    if not ALLOW_REMOTE_MEDIA:
        st.caption(
            "Remote media is not fetched. Set MMI_VIEWER_ALLOW_REMOTE=1 to render "
            "these inline."
        )


def get_produced(r: dict) -> list[str]:
    """Derive produced modalities from per_modality scores."""
    produced = []
    for mod, ms in r.get("per_modality", {}).items():
        if (
            ms.get("detected_native")
            or ms.get("detected_via_url")
            or ms.get("detected_via_judge")
        ):
            produced.append(mod)
    return sorted(produced)


# ── UI ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="MMI Result Viewer", page_icon="🔍", layout="wide")
st.title("🔍 MMI Result Viewer")

prompts = load_prompts()
configs = discover_configs()

if not configs:
    st.error(f"No result directories found under `{RESULTS_ROOT}`")
    st.stop()

# ── sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")

    selected_config = st.selectbox("Config", configs)

    runs = discover_runs(selected_config)
    if not runs:
        st.error("No result files in this config directory.")
        st.stop()

    selected_label = st.selectbox("Run", list(runs.keys()))
    results = load_results(runs[selected_label])

    filter_status = st.radio(
        "Result", ["All", "Pass only", "Fail only"], horizontal=True
    )

    all_input_modalities = sorted(
        {
            m
            for r in results
            for m in get_prompt(prompts, r.get("prompt_id", ""))["_input_modalities"]
        }
    )
    filter_input_modalities = st.multiselect(
        "Input modality", all_input_modalities, default=all_input_modalities
    )

    all_output_modalities = sorted(
        {m for r in results for m in r.get("expected_modalities", [])}
    )
    filter_output_modalities = st.multiselect(
        "Expected output modality",
        all_output_modalities,
        default=all_output_modalities,
    )

    search_query = st.text_input("Search prompt text or ID")

# ── apply filters ─────────────────────────────────────────────────────

filtered: list[dict] = []
for r in results:
    all_pass = r.get("all_pass_lenient", False)
    if filter_status == "Pass only" and not all_pass:
        continue
    if filter_status == "Fail only" and all_pass:
        continue
    if filter_output_modalities and not any(
        m in filter_output_modalities for m in r.get("expected_modalities", [])
    ):
        continue
    if filter_input_modalities:
        p = get_prompt(prompts, r.get("prompt_id", ""))
        if not any(m in filter_input_modalities for m in p["_input_modalities"]):
            continue
    if search_query:
        pid = r.get("prompt_id", "")
        p = get_prompt(prompts, pid)
        q = search_query.lower()
        if q not in pid.lower() and q not in p.get("prompt_text", "").lower():
            continue
    filtered.append(r)

# ── summary bar ───────────────────────────────────────────────────────

model_name = extract_model_name(selected_label)
summary = load_summary(selected_config, model_name)

total = len(filtered)
passed = sum(1 for r in filtered if r.get("all_pass_lenient"))
failed = total - passed

col1, col2, col3 = st.columns(3)
col1.metric("Showing", f"{total} results")
col2.metric("Pass", f"{passed}", delta=f"{passed / total * 100:.0f}%" if total else "–")
col3.metric(
    "Fail",
    f"{failed}",
    delta=f"{failed / total * 100:.0f}%" if total else "–",
    delta_color="inverse",
)

if summary:
    st.markdown(
        "**Per-modality pass rates:** "
        + "  ·  ".join(
            format_rate(k, v) for k, v in summary.items() if not k.startswith("_")
        )
    )
    overall = summary.get("_overall", 0)
    overall_strict = summary.get("_overall_strict", 0)
    if isinstance(overall, (int, float)):
        st.markdown(
            f"**Overall (lenient):** **{overall:.0%}**"
            f"  ·  **Overall (strict):** **{overall_strict:.0%}**"
        )

st.divider()

# ── result cards ──────────────────────────────────────────────────────

st.caption(f"{total} results")

for r in filtered:
    pid = r["prompt_id"]
    p = get_prompt(prompts, pid)
    prompt_text = p.get("prompt_text", "")
    input_mods = p["_input_modalities"]
    input_files = p["_input_files"]
    all_ok = r.get("all_pass_lenient", False)

    expected_mods = r.get("expected_modalities", [])
    produced = get_produced(r)

    input_icons = " ".join(MODALITY_ICONS.get(m, "❓") for m in input_mods)
    output_icons = " ".join(MODALITY_ICONS.get(m, "❓") for m in expected_mods)
    detected_icons = " ".join(MODALITY_ICONS.get(m, "❓") for m in produced)

    header_text = f"{pass_fail(all_ok)}  `{pid}`"
    if input_icons or output_icons:
        header_text += f"  {input_icons} → {output_icons}"
    if detected_icons:
        header_text += f"  (detected: {detected_icons})"
    if prompt_text:
        snippet = prompt_text[:100] + ("…" if len(prompt_text) > 100 else "")
        header_text += f" — {snippet}"

    with st.expander(header_text, expanded=False):
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown(
                "**Input modalities:** "
                + (
                    "  ".join(modality_badge(m) for m in input_mods)
                    if input_mods
                    else "_(none)_"
                )
            )
        with info_col2:
            st.markdown(
                "**Expected output:** "
                + (
                    "  ".join(modality_badge(m) for m in expected_mods)
                    if expected_mods
                    else "_(none)_"
                )
            )

        if input_files:
            render_input_attachments(input_files)

        if prompt_text:
            st.markdown(f"**Prompt:** {prompt_text}")

        if r.get("judge_used"):
            st.markdown("🧑‍⚖️ *LLM Judge was used for this result*")
        per_modality = r.get("per_modality", {})

        score_cols = st.columns(len(expected_mods)) if expected_mods else []
        for i, mod in enumerate(expected_mods):
            ms = per_modality.get(mod, {})
            ok = ms.get("pass_lenient", False)
            if ms.get("detected_native"):
                source = "native"
            elif ms.get("detected_via_url"):
                source = "via URL"
            elif ms.get("detected_via_judge"):
                source = "via judge"
            else:
                source = "missing"
            score_cols[i].markdown(
                f"**{modality_badge(mod)}**\n\n{pass_fail(ok)} {source}"
            )

        st.markdown(
            f"**Produced modalities:** "
            f"{', '.join(modality_badge(m) for m in produced) if produced else '_(none)_'}"
        )

        if r.get("judge_used"):
            st.markdown("🧑‍⚖️ *LLM Judge was used for this result*")

        raw = r.get("raw_response")
        if raw:
            if not isinstance(raw, str):
                raw = json.dumps(raw, indent=2, default=str)
            st.markdown("---")
            st.markdown("**Raw response**")
            render_response_media(raw)
            st.code(raw, language=None)

st.caption("MMI Harness — Result Viewer")
