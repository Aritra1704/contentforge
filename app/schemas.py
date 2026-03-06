"""Shared request and response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

BackendName = Literal["ollama", "groq"]
EmojiPolicy = Literal["none", "light", "expressive"]
ToneStyle = Literal["minimal", "poetic", "conversational", "witty", "inspirational"]
OutputFormat = Literal["lines", "numbered"]
OutputSpecFormat = Literal["one_liner", "paragraph", "one_page", "pros_cons", "verse", "story"]
WinnerSource = Literal["baseline", "judge_openai", "judge_ollama"]

DEFAULT_AVOID_PHRASES = [
    "new week",
    "rise and shine",
    "inner strength",
    "you got this",
    "make it happen",
    "shine bright",
    "positive vibes",
]


class OutputLengthSpec(BaseModel):
    """Word-length constraints for generated output."""

    min_words: int | None = Field(default=None, ge=1, le=5000)
    max_words: int | None = Field(default=None, ge=1, le=5000)
    target_words: int | None = Field(default=None, ge=1, le=5000)

    @model_validator(mode="after")
    def validate_bounds(self) -> "OutputLengthSpec":
        """Ensure min/target/max constraints remain coherent."""

        if self.min_words is not None and self.max_words is not None and self.min_words > self.max_words:
            raise ValueError("length.min_words cannot be greater than length.max_words")
        return self


class OutputStructureSpec(BaseModel):
    """Structural constraints for generated output."""

    items: int | None = Field(default=None, ge=1, le=100)
    sections: list[str] | None = None
    max_lines: int | None = Field(default=None, ge=1, le=200)
    no_lists: bool | None = None
    no_numbering: bool | None = None
    max_words_per_line: int | None = Field(default=None, ge=1, le=64)

    @field_validator("sections")
    @classmethod
    def normalize_sections(cls, value: list[str] | None) -> list[str] | None:
        """Normalize section names and drop blanks/duplicates."""

        if value is None:
            return None

        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            candidate = item.strip()
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(candidate)
        return normalized or None


class OutputSpec(BaseModel):
    """Unified output controls shared by generation endpoints."""

    format: OutputSpecFormat = "one_liner"
    length: OutputLengthSpec = Field(default_factory=OutputLengthSpec)
    structure: OutputStructureSpec = Field(default_factory=OutputStructureSpec)


def _payload_get(payload: BaseModel | dict[str, Any], key: str) -> Any:
    """Read one field from either a model instance or plain dict payload."""

    if isinstance(payload, BaseModel):
        return getattr(payload, key, None)
    return payload.get(key)


def _payload_has(payload: BaseModel | dict[str, Any], key: str) -> bool:
    """Return whether a field was explicitly supplied."""

    if isinstance(payload, BaseModel):
        return key in payload.model_fields_set
    return key in payload


def _output_spec_defaults(format_name: OutputSpecFormat) -> OutputSpec:
    """Return defaults for one output format."""

    if format_name == "paragraph":
        return OutputSpec(
            format="paragraph",
            length=OutputLengthSpec(min_words=60, max_words=110, target_words=80),
            structure=OutputStructureSpec(no_lists=True, no_numbering=True),
        )
    if format_name == "one_page":
        return OutputSpec(
            format="one_page",
            length=OutputLengthSpec(min_words=180, max_words=320, target_words=250),
            structure=OutputStructureSpec(no_lists=True, no_numbering=True),
        )
    if format_name == "pros_cons":
        return OutputSpec(
            format="pros_cons",
            length=OutputLengthSpec(),
            structure=OutputStructureSpec(
                items=4,
                sections=["Pros", "Cons"],
                no_numbering=True,
            ),
        )
    if format_name == "verse":
        return OutputSpec(
            format="verse",
            length=OutputLengthSpec(),
            structure=OutputStructureSpec(
                items=8,
                max_lines=12,
                no_lists=True,
                no_numbering=True,
                max_words_per_line=8,
            ),
        )
    if format_name == "story":
        return OutputSpec(
            format="story",
            length=OutputLengthSpec(min_words=350, max_words=600, target_words=450),
            structure=OutputStructureSpec(
                sections=["Act I", "Act II", "Act III"],
                no_lists=True,
                no_numbering=True,
            ),
        )
    return OutputSpec(
        format="one_liner",
        length=OutputLengthSpec(target_words=10),
        structure=OutputStructureSpec(items=3),
    )


def _merge_output_spec(spec: OutputSpec, defaults: OutputSpec) -> OutputSpec:
    """Fill missing OutputSpec fields from defaults without overwriting explicit values."""

    merged = spec.model_copy(deep=True)

    if merged.length.min_words is None:
        merged.length.min_words = defaults.length.min_words
    if merged.length.max_words is None:
        merged.length.max_words = defaults.length.max_words
    if merged.length.target_words is None:
        merged.length.target_words = defaults.length.target_words

    if merged.structure.items is None:
        merged.structure.items = defaults.structure.items
    if not merged.structure.sections and defaults.structure.sections:
        merged.structure.sections = list(defaults.structure.sections)
    if merged.structure.max_lines is None:
        merged.structure.max_lines = defaults.structure.max_lines
    if merged.structure.no_lists is None:
        merged.structure.no_lists = defaults.structure.no_lists
    if merged.structure.no_numbering is None:
        merged.structure.no_numbering = defaults.structure.no_numbering
    if merged.structure.max_words_per_line is None:
        merged.structure.max_words_per_line = defaults.structure.max_words_per_line

    return merged


def normalize_output_spec(payload: BaseModel | dict[str, Any]) -> OutputSpec:
    """Build one normalized OutputSpec from new and legacy payload fields."""

    raw_spec = _payload_get(payload, "output_spec")
    has_output_spec = _payload_has(payload, "output_spec") and raw_spec is not None

    if isinstance(raw_spec, OutputSpec):
        normalized = raw_spec.model_copy(deep=True)
    elif raw_spec is None:
        normalized = OutputSpec()
    else:
        normalized = OutputSpec.model_validate(raw_spec)

    if not has_output_spec:
        legacy_count = _payload_get(payload, "count")
        if legacy_count is not None and normalized.format in {"one_liner", "pros_cons"}:
            normalized.structure.items = int(legacy_count)

        legacy_max_words = _payload_get(payload, "max_words")
        if legacy_max_words is not None:
            normalized.length.max_words = int(legacy_max_words)

        legacy_min_words = _payload_get(payload, "min_words")
        if legacy_min_words is not None:
            normalized.length.min_words = int(legacy_min_words)

        legacy_output_format = _payload_get(payload, "output_format")
        if legacy_output_format is not None and normalized.format == "one_liner":
            normalized.structure.no_numbering = legacy_output_format == "lines"

    normalized = _merge_output_spec(normalized, _output_spec_defaults(normalized.format))

    return OutputSpec.model_validate(normalized.model_dump(mode="python", exclude_none=True))


class GenerateSingleRequest(BaseModel):
    """Request payload for one stateless content generation call."""

    theme_name: str = Field(min_length=1)
    tone_funny_pct: int = Field(ge=0, le=100)
    tone_emotion_pct: int = Field(ge=0, le=100)
    prompt_keywords: list[str] = Field(default_factory=list)
    visual_style: str = Field(min_length=1)
    backend: BackendName
    model: str = Field(min_length=1)
    count: int = Field(default=3, ge=1, le=10)
    max_tokens: int = Field(default=300, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0, le=2)
    max_words: int = Field(default=16, ge=1, le=5000)
    min_words: int | None = Field(default=None, ge=1, le=5000)
    emoji_policy: EmojiPolicy = "none"
    tone_style: ToneStyle = "conversational"
    audience: str = "general"
    avoid_cliches: bool = True
    avoid_phrases: list[str] = Field(default_factory=lambda: list(DEFAULT_AVOID_PHRASES))
    output_format: OutputFormat = "numbered"
    output_spec: OutputSpec | None = None
    trace_id: str | None = None
    seed: int | None = None

    @field_validator("prompt_keywords")
    @classmethod
    def normalize_keywords(cls, value: list[str]) -> list[str]:
        """Remove blank keywords before prompt construction."""

        return [item.strip() for item in value if item.strip()]

    @field_validator("model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        """Normalize the model name before backend validation."""

        return value.strip()

    @field_validator("audience")
    @classmethod
    def strip_audience(cls, value: str) -> str:
        """Normalize audience text and enforce a non-empty value."""

        value = value.strip()
        return value or "general"

    @field_validator("avoid_phrases")
    @classmethod
    def normalize_avoid_phrases(cls, value: list[str]) -> list[str]:
        """Normalize and deduplicate avoid-phrase values."""

        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            candidate = item.strip()
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(candidate)
        return normalized

    @field_validator("trace_id")
    @classmethod
    def strip_trace_id(cls, value: str | None) -> str | None:
        """Normalize optional trace IDs."""

        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def apply_output_spec(self) -> "GenerateSingleRequest":
        """Populate normalized output constraints for downstream components."""

        self.output_spec = normalize_output_spec(self)
        return self


class ResponseMeta(BaseModel):
    """Metadata attached to successful generation responses."""

    latency_ms: int | None = None
    request_id: str
    trace_id: str | None = None
    busy: bool = False
    applied_settings: dict[str, Any] | None = None


class ErrorBody(BaseModel):
    """Structured error payload returned by the service."""

    error_type: str
    message: str
    backend: BackendName | str | None = None
    model: str | None = None
    http_status: int | None = None
    retry_after_ms: int | None = None
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Error response body returned by the service."""

    ok: Literal[False]
    error: ErrorBody
    meta: ResponseMeta


class ProsConsStructuredOutput(BaseModel):
    """Structured response payload for pros/cons outputs."""

    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)


class QualityScore(BaseModel):
    """Quality-first scoring breakdown for one generated output."""

    format_compliance: int = Field(ge=0, le=30)
    tone_alignment: int = Field(ge=0, le=20)
    originality: int = Field(ge=0, le=20)
    clarity_coherence: int = Field(ge=0, le=20)
    policy_cleanliness: int = Field(ge=0, le=10)
    total: int = Field(ge=0, le=100)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class GenerateSingleResponse(BaseModel):
    """Successful response for POST /generate/single."""

    ok: Literal[True]
    backend: BackendName
    model: str
    items: list[str] = Field(default_factory=list)
    raw_text: str | None = None
    structured_output: ProsConsStructuredOutput | None = None
    meta: ResponseMeta
    errors: ErrorBody | None = None


class CompareModelTarget(BaseModel):
    """One backend/model target in a compare request."""

    backend: BackendName
    model: str = Field(min_length=1)

    @field_validator("model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        """Normalize the target model name."""

        return value.strip()


class GenerateCompareModelsRequest(BaseModel):
    """Shared prompt plus multiple backend/model targets."""

    theme_name: str = Field(min_length=1)
    tone_funny_pct: int = Field(ge=0, le=100)
    tone_emotion_pct: int = Field(ge=0, le=100)
    prompt_keywords: list[str] = Field(default_factory=list)
    visual_style: str = Field(min_length=1)
    targets: list[CompareModelTarget] = Field(min_length=1)
    count: int = Field(default=3, ge=1, le=10)
    max_tokens: int = Field(default=300, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0, le=2)
    max_words: int = Field(default=16, ge=1, le=5000)
    min_words: int | None = Field(default=None, ge=1, le=5000)
    emoji_policy: EmojiPolicy = "none"
    tone_style: ToneStyle = "conversational"
    audience: str = "general"
    avoid_cliches: bool = True
    avoid_phrases: list[str] = Field(default_factory=lambda: list(DEFAULT_AVOID_PHRASES))
    output_format: OutputFormat = "numbered"
    output_spec: OutputSpec | None = None
    trace_id: str | None = None
    seed: int | None = None

    @field_validator("prompt_keywords")
    @classmethod
    def normalize_keywords(cls, value: list[str]) -> list[str]:
        """Remove blank keywords before prompt construction."""

        return [item.strip() for item in value if item.strip()]

    @field_validator("trace_id")
    @classmethod
    def strip_trace_id(cls, value: str | None) -> str | None:
        """Normalize optional trace IDs."""

        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("audience")
    @classmethod
    def strip_audience(cls, value: str) -> str:
        """Normalize audience text and enforce a non-empty value."""

        value = value.strip()
        return value or "general"

    @field_validator("avoid_phrases")
    @classmethod
    def normalize_avoid_phrases(cls, value: list[str]) -> list[str]:
        """Normalize and deduplicate avoid-phrase values."""

        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            candidate = item.strip()
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(candidate)
        return normalized

    @model_validator(mode="after")
    def apply_output_spec(self) -> "GenerateCompareModelsRequest":
        """Populate normalized output constraints for downstream components."""

        self.output_spec = normalize_output_spec(self)
        return self


class CompareModelResult(BaseModel):
    """One model result for compare-models responses."""

    ok: bool
    backend: BackendName
    model: str
    latency_ms: int | None = None
    items: list[str] = Field(default_factory=list)
    raw_text: str | None = None
    structured_output: ProsConsStructuredOutput | None = None
    quality: QualityScore | None = None
    error: ErrorBody | None = None


class CompareModelsWinner(BaseModel):
    """Winner metadata for quality-first compare-models runs."""

    backend: BackendName
    model: str
    total_score: int = Field(ge=0, le=100)


class JudgeCandidateScore(BaseModel):
    """One per-candidate judge score breakdown."""

    format_compliance: int = Field(ge=0, le=30)
    tone_alignment: int = Field(ge=0, le=20)
    originality: int = Field(ge=0, le=20)
    clarity_coherence: int = Field(ge=0, le=20)
    policy_cleanliness: int = Field(ge=0, le=10)
    total: int = Field(ge=0, le=100)
    reasons: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    """Structured winner/ranking decision from LLM judge."""

    winner_key: str
    ranking: list[str] = Field(default_factory=list)
    scores: dict[str, JudgeCandidateScore] = Field(default_factory=dict)


class GenerateCompareModelsResponse(BaseModel):
    """Response for compare-models requests."""

    ok: bool
    results: list[CompareModelResult]
    winner: CompareModelsWinner | None = None
    winner_source: WinnerSource = "baseline"
    judge_result: JudgeResult | None = None
    # Backward-compatible field for existing clients.
    judge_json: dict[str, Any] | None = None
    judge_reason: str | None = None
    meta: ResponseMeta


class QualityRunHistoryItem(BaseModel):
    """Stored quality-memory row returned by /quality/history."""

    run_id: str
    created_at: datetime
    theme_name: str
    keywords: list[str] = Field(default_factory=list)
    tone_config: dict[str, Any] = Field(default_factory=dict)
    output_spec: dict[str, Any] = Field(default_factory=dict)
    backend: str
    model: str
    output_text: str
    quality_score_json: dict[str, Any] = Field(default_factory=dict)
    judge_json: dict[str, Any] | None = None
    detected_cliches: list[str] = Field(default_factory=list)
    repetition_flags: list[str] = Field(default_factory=list)
    json_leak_flag: bool = False


class QualityHistoryResponse(BaseModel):
    """Response for GET /quality/history."""

    ok: Literal[True]
    runs: list[QualityRunHistoryItem]


class OllamaModelCatalog(BaseModel):
    """Ollama-discovered chat and embedding model lists."""

    chat_models: list[str]
    embedding_models: list[str]


class ModelDiscoveryResponse(BaseModel):
    """Successful response for GET /models."""

    ok: Literal[True]
    ollama: OllamaModelCatalog


class HealthResponse(BaseModel):
    """Successful response for GET /health."""

    ok: Literal[True]
    service: str
    version: str
    busy: bool
    ollama_reachable: bool
