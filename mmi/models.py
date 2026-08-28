# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Data models for the MMI evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RubricCriterion:
    """A rubric criterion tied to the output modality it grades."""

    id: str
    criterion: str
    modality: str


@dataclass
class EvalPrompt:
    """A single evaluation prompt from the MMI dataset."""

    prompt_id: str
    prompt_text: str
    input_modalities: list[str]
    output_modalities: list[str]
    input_files: list[str] = field(default_factory=list)
    rubric_criteria: list[RubricCriterion] = field(default_factory=list)


@dataclass
class ModalityDetection:
    """Detection result for a single modality."""

    detected_native: bool = False
    detected_via_url: bool = False
    native_evidence: str = ""
    url_evidence: str = ""


@dataclass
class DetectionResult:
    """Complete structural detection output for a single response."""

    modalities: dict[str, ModalityDetection]
    text_content: str = ""
    urls_found: list[str] = field(default_factory=list)
    is_error: bool = False
    error_message: str = ""
    error_type: str = ""


@dataclass
class JudgeResult:
    """Output from the LLM judge (Layer 2 detection).

    The judge acts as a safety net for false negatives in structural
    detection.  ``detected_modalities`` lists modalities the judge
    believes the model actually produced but the parser missed.
    """

    detected_modalities: list[str] = field(default_factory=list)
    reasoning: str = ""
    judge_model: str = ""
    judge_raw_response: str = ""


@dataclass
class RequestRecord:
    """Provider-facing request metadata for one inference call.

    This intentionally records text/config and input filenames only. The
    immutable dataset remains the source of truth for input asset bytes.
    """

    provider: str
    api: str
    model: str
    system_prompt: str = ""
    user_prompt: str = ""
    input_files: list[str] = field(default_factory=list)
    input_modalities: list[str] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    response_modalities: list[str] = field(default_factory=list)
    max_output_tokens: int | None = None
    provider_request: dict[str, Any] = field(default_factory=dict)


@dataclass
class CapturedAsset:
    """A model-produced asset captured during response processing.

    Three concepts are kept separate on purpose:

    - ``delivery`` — *how it arrived* (structural provenance). Never inferred
      from a hostname.
    - ``capture_status`` — *whether we hold the bytes*.
    - the scoring class — *what the metric makes of it* — which is derived from
      the two above by :func:`mmi.detection.scoring_class` and deliberately not
      stored here, so a provider cannot assert it.
    """

    asset_id: str
    prompt_id: str
    modality: str
    source_type: str
    delivery: str = ""
    mime_type: str = ""
    # Only meaningful for genuinely opaque MIME. The shared table wins wherever
    # it can classify, so this can never mint credit.
    modality_hint: str = ""
    local_path: str = ""
    source_url: str = ""
    sha256: str = ""
    size_bytes: int = 0
    capture_status: str = "captured"
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    """Normalized trace for a neutral MMI tool call."""

    tool_name: str
    provider_call_id: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    error: str = ""
    produced_asset_ids: list[str] = field(default_factory=list)


@dataclass
class ProviderResponse:
    """The response from a provider for a single prompt.

    This is the whole provider contract. A provider extracts; it does not
    adjudicate. ``response_text`` is user-visible prose only — never tool
    traces — and ``raw_response`` must be JSON-serializable, never ``str()``.
    """

    prompt_id: str
    run_name: str
    provider: str
    model: str
    response_text: str = ""
    raw_response: Any = None
    error: str | None = None
    is_error: bool = False
    error_type: str = ""
    request: RequestRecord | None = None
    output_assets: list[CapturedAsset] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    @property
    def detection(self) -> DetectionResult:
        """Adjudicate this response through the single shared detector."""
        from .detection import build_detection_result

        result = build_detection_result(self.response_text, self.output_assets)
        result.is_error = self.is_error
        result.error_message = self.error or ""
        result.error_type = self.error_type
        return result


@dataclass
class ModalityScore:
    """Scored result for a single expected modality."""

    detected_native: bool = False
    detected_via_url: bool = False
    detected_via_judge: bool = False
    pass_strict: bool = False
    pass_lenient: bool = False


@dataclass
class EvalResult:
    """Scored result comparing expected vs produced modalities."""

    prompt_id: str
    run_name: str
    provider: str
    model: str
    expected_modalities: list[str]
    per_modality: dict[str, ModalityScore]
    all_pass_lenient: bool
    all_pass_strict: bool
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    precision_strict: float = 0.0
    recall_strict: float = 0.0
    f1_strict: float = 0.0
    judge_used: bool = False
    #: Which judge produced the Layer-2 evidence, and why. Recorded per row so
    #: a merged or rejudged file cannot hide a change of judge.
    judge_model: str = ""
    judge_reasoning: str = ""
    is_error: bool = False
    error_message: str = ""
    error_type: str = ""
    raw_response: Any = None
    response_text: str = ""
    request: dict[str, Any] | None = None
    output_assets: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    rubric_used: bool = False
    #: The MMI Value for this prompt: the mean over the prompt's output
    #: modalities of the mean criterion score within each. Averaging within a
    #: modality first keeps every requested modality equally weighted, so a
    #: modality cannot gain influence by carrying more criteria.
    rubric_score: float | None = None
    rubric_grades: list[dict[str, Any]] = field(default_factory=list)
    #: The within-modality means that ``rubric_score`` averages, kept so the
    #: headline number can be audited from the row itself.
    rubric_score_by_modality: dict[str, float] = field(default_factory=dict)
    #: The paper's metric: 1 per modality only if every rubric for that
    #: modality was marked correct (results.tex). Reported alongside the mean,
    #: never instead of it.
    rubric_binary_by_modality: dict[str, int] = field(default_factory=dict)
    #: Criteria whose zero came from the judge failing rather than the model.
    #: They still score zero — anything else would inflate results — but they
    #: stay countable so a degraded run cannot pass as a bad model.
    rubric_evaluator_errors: int = 0
    rubric_judge_model: str = ""

    @property
    def produced_modalities(self) -> list[str]:
        """Modalities detected (native, via URL, or via judge)."""
        return sorted(
            m
            for m, s in self.per_modality.items()
            if s.detected_native or s.detected_via_url or s.detected_via_judge
        )
