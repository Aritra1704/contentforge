"""Candidate-level ranking for compare-model content outputs."""

from __future__ import annotations

from difflib import SequenceMatcher
import re

from app.llm import GeneratedOutput
from app.quality import score_quality
from app.schemas import (
    CandidateRankingSummary,
    CompareModelResult,
    GenerateCompareModelsRequest,
    GenerateSingleRequest,
    JudgeResult,
    OutputSpec,
    ProsConsStructuredOutput,
    QualityScore,
    RankedTextCandidate,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
MAX_SHORTLIST_CANDIDATES = 5
LONG_FORM_FORMATS = {"paragraph", "one_page", "story"}
DIVERSITY_BONUS_CAP = 3.0
ABRUPT_ENDINGS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "for",
    "from",
    "if",
    "in",
    "into",
    "of",
    "on",
    "or",
    "our",
    "the",
    "their",
    "this",
    "to",
    "with",
    "your",
}

_REASON_TEXT = {
    "judge_preferred_model": "comes from the judge-preferred model run",
    "top_model_run": "comes from a top-ranked model run",
    "high_quality_model_run": "comes from a strong model run",
    "strong_task_fit": "matches the requested theme and tone well",
    "complete_output": "reads as complete and ready to use",
    "distinct_phrasing": "uses distinct phrasing",
    "human_tone": "sounds human and emotionally believable",
    "clear_flow": "reads clearly and smoothly",
    "overall_quality": "landed best on overall candidate quality",
}


def build_ranked_candidates(
    payload: GenerateCompareModelsRequest,
    results: list[CompareModelResult],
    *,
    judge_result: JudgeResult | None = None,
    judge_candidate_map: dict[str, CompareModelResult] | None = None,
    shortlist_max: int = MAX_SHORTLIST_CANDIDATES,
) -> tuple[list[RankedTextCandidate], list[RankedTextCandidate], CandidateRankingSummary]:
    """Return ranked, deduplicated candidates and shortlist entries."""

    summary = CandidateRankingSummary()
    judge_candidate_map = judge_candidate_map or {}
    model_rank_lookup = _build_model_rank_lookup(
        results,
        judge_result=judge_result,
        judge_candidate_map=judge_candidate_map,
    )

    preliminary: list[dict[str, object]] = []
    for result in results:
        if not result.ok:
            continue

        candidate_request = _build_candidate_request(payload, result)
        model_score = result.quality.total if result.quality is not None else 0
        model_rank = model_rank_lookup.get(_result_key(result))

        for index, text in enumerate(_extract_candidate_texts(payload, result)):
            summary.total_candidates_seen += 1
            if _is_incomplete_candidate_text(payload, text):
                summary.rejected_incomplete_count += 1
                continue
            candidate_output = _build_candidate_output(candidate_request, text)
            candidate_quality, is_valid = score_quality(candidate_request, candidate_output)
            if not is_valid:
                if _is_incomplete_quality(candidate_quality):
                    summary.rejected_incomplete_count += 1
                else:
                    summary.rejected_invalid_count += 1
                continue

            model_bonus = _model_rank_bonus(model_rank)
            diversity_bonus = _diversity_bonus(text)
            final_score = round(candidate_quality.total + model_bonus + diversity_bonus, 4)
            preliminary.append(
                {
                    "backend": result.backend,
                    "model": result.model,
                    "text": text,
                    "source_item_index": index,
                    "score": final_score,
                    "diversity_bonus": diversity_bonus,
                    "model_score": model_score,
                    "model_rank": model_rank,
                    "quality": candidate_quality,
                    "normalized_text": _normalize_text(text),
                }
            )

    preliminary.sort(
        key=lambda item: (
            float(item["score"]),
            int(item["model_score"]),
            ((item["quality"]).task_fit if isinstance(item["quality"], QualityScore) else 0),
            ((item["quality"]).completeness if isinstance(item["quality"], QualityScore) else 0),
            str(item["model"]),
            str(item["text"]),
        ),
        reverse=True,
    )

    ranked: list[RankedTextCandidate] = []
    kept_normalized: list[str] = []
    for item in preliminary:
        normalized_text = str(item["normalized_text"])
        if any(_is_near_duplicate(normalized_text, existing) for existing in kept_normalized):
            summary.rejected_duplicate_count += 1
            continue

        quality = item["quality"]
        assert isinstance(quality, QualityScore)
        model_rank = item["model_rank"] if isinstance(item["model_rank"], int) else None
        reason_codes = _reason_codes_for_candidate(
            quality,
            model_rank=model_rank,
            judge_result=judge_result,
        )
        ranked.append(
            RankedTextCandidate(
                backend=str(item["backend"]),
                model=str(item["model"]),
                text=str(item["text"]),
                source_item_index=int(item["source_item_index"]),
                rank=len(ranked) + 1,
                score=float(item["score"]),
                model_score=int(item["model_score"]),
                reason=_summarize_reason_codes(reason_codes),
                reason_codes=reason_codes,
            )
        )
        kept_normalized.append(normalized_text)

    shortlist_count = min(max(shortlist_max, 1), len(ranked))
    shortlist = [candidate.model_copy(deep=True) for candidate in ranked[:shortlist_count]]

    summary.ranked_candidate_count = len(ranked)
    summary.shortlisted_count = len(shortlist)
    return ranked, shortlist, summary


def _build_candidate_request(
    payload: GenerateCompareModelsRequest,
    result: CompareModelResult,
) -> GenerateSingleRequest:
    """Build one single-request payload for candidate-level quality scoring."""

    spec = payload.output_spec.model_copy(deep=True) if payload.output_spec is not None else OutputSpec()
    if spec.format == "one_liner":
        spec.structure.items = 1

    return GenerateSingleRequest(
        app_id=payload.app_id,
        content_type=payload.content_type,
        theme_name=payload.theme_name,
        tone_funny_pct=payload.tone_funny_pct,
        tone_emotion_pct=payload.tone_emotion_pct,
        prompt_keywords=payload.prompt_keywords,
        visual_style=payload.visual_style,
        backend=result.backend,
        model=result.model,
        count=1,
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
        max_words=payload.max_words,
        min_words=payload.min_words,
        emoji_policy=payload.emoji_policy,
        tone_style=payload.tone_style,
        audience=payload.audience,
        cultural_context=payload.cultural_context,
        avoid_cliches=payload.avoid_cliches,
        avoid_phrases=payload.avoid_phrases,
        output_format=payload.output_format,
        output_spec=spec,
        creative_brief=payload.creative_brief.model_copy(deep=True) if payload.creative_brief is not None else None,
        trace_id=payload.trace_id,
        seed=payload.seed,
    )


def _extract_candidate_texts(
    payload: GenerateCompareModelsRequest,
    result: CompareModelResult,
) -> list[str]:
    """Explode one compare-model result into candidate texts."""

    spec = payload.output_spec or OutputSpec()
    if spec.format == "pros_cons":
        structured = result.structured_output or ProsConsStructuredOutput()
        pros = "\n".join(f"- {item}" for item in structured.pros if str(item).strip())
        cons = "\n".join(f"- {item}" for item in structured.cons if str(item).strip())
        text = f"Pros:\n{pros}\nCons:\n{cons}".strip()
        return [text] if text else []

    if spec.format in LONG_FORM_FORMATS:
        text = str(result.raw_text or "").strip()
        return [text] if text else []

    items = [str(item or "").strip() for item in result.items if str(item or "").strip()]
    if items:
        return items

    text = str(result.raw_text or "").strip()
    return [text] if text else []


def _build_candidate_output(payload: GenerateSingleRequest, text: str) -> GeneratedOutput:
    """Wrap one text candidate in the output shape expected by `score_quality`."""

    if (payload.output_spec or OutputSpec()).format == "pros_cons":
        return GeneratedOutput(items=[text], raw_text=text, structured_output=None)
    return GeneratedOutput(items=[text], raw_text=text, structured_output=None)


def _is_incomplete_candidate_text(payload: GenerateCompareModelsRequest, text: str) -> bool:
    """Return whether a short candidate is clearly incomplete before shortlist ranking."""

    spec = payload.output_spec or OutputSpec()
    if spec.format != "one_liner":
        return False

    tokens = TOKEN_RE.findall(text.lower())
    target_words = spec.length.target_words or payload.max_words or 16
    min_words = max(4, min(8, max(4, int(target_words) // 2)))
    stripped = text.strip()
    if not stripped or len(tokens) < min_words:
        return True
    if stripped.endswith("...") or stripped.endswith(".."):
        return True
    if stripped[-1] not in ".!?":
        return True
    return bool(tokens and tokens[-1] in ABRUPT_ENDINGS)


def _build_model_rank_lookup(
    results: list[CompareModelResult],
    *,
    judge_result: JudgeResult | None,
    judge_candidate_map: dict[str, CompareModelResult],
) -> dict[str, int]:
    """Return model rank positions from judge output first, baseline quality second."""

    ranks: dict[str, int] = {}
    next_rank = 0

    if judge_result is not None:
        for candidate_key in judge_result.ranking:
            mapped = judge_candidate_map.get(candidate_key)
            if mapped is None:
                continue
            key = _result_key(mapped)
            if key in ranks:
                continue
            ranks[key] = next_rank
            next_rank += 1

    valid_results = [item for item in results if item.ok]
    valid_results.sort(
        key=lambda item: (
            item.quality.total if item.quality is not None else 0,
            item.quality.task_fit if item.quality is not None else 0,
            item.quality.completeness if item.quality is not None else 0,
            item.model,
        ),
        reverse=True,
    )
    for item in valid_results:
        key = _result_key(item)
        if key in ranks:
            continue
        ranks[key] = next_rank
        next_rank += 1
    return ranks


def _result_key(result: CompareModelResult) -> str:
    """Return stable backend:model key for ranking tie-breaks."""

    return f"{result.backend}:{result.model}"


def _model_rank_bonus(model_rank: int | None) -> float:
    """Return a small model-level bonus without dominating candidate quality."""

    if model_rank is None:
        return 0.0
    return max(0.0, round(3.0 - (model_rank * 0.75), 4))


def _diversity_bonus(text: str) -> float:
    """Return a small lexical-diversity bonus for shortlist ranking."""

    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    if len(tokens) < 5:
        return 0.0
    ratio = len(set(tokens)) / len(tokens)
    if ratio < 0.55:
        return 0.0
    return round(min(DIVERSITY_BONUS_CAP, ratio * 2.5), 4)


def _reason_codes_for_candidate(
    quality: QualityScore,
    *,
    model_rank: int | None,
    judge_result: JudgeResult | None,
) -> list[str]:
    """Return stable reason codes for shortlist display."""

    codes: list[str] = []
    if judge_result is not None and model_rank == 0:
        codes.append("judge_preferred_model")
    elif model_rank == 0:
        codes.append("top_model_run")
    elif model_rank is not None and model_rank <= 2:
        codes.append("high_quality_model_run")

    if quality.task_fit >= 18:
        codes.append("strong_task_fit")
    if quality.completeness >= 12 and quality.incomplete_ending_penalty == 0:
        codes.append("complete_output")
    if quality.originality >= 14 and quality.overused_pattern_penalty == 0:
        codes.append("distinct_phrasing")
    if quality.emotional_authenticity >= 14 and quality.robotic_tone_penalty == 0:
        codes.append("human_tone")
    if quality.clarity_and_flow >= 8:
        codes.append("clear_flow")
    if not codes:
        codes.append("overall_quality")
    return codes[:4]


def _summarize_reason_codes(reason_codes: list[str]) -> str:
    """Render a concise shortlist-ready reason sentence."""

    if not reason_codes:
        return "Ranked by overall candidate quality."

    phrases = [_REASON_TEXT[code] for code in reason_codes if code in _REASON_TEXT]
    if not phrases:
        return "Ranked by overall candidate quality."
    if len(phrases) == 1:
        return f"This candidate {phrases[0]}."
    if len(phrases) == 2:
        return f"This candidate {phrases[0]} and {phrases[1]}."
    return f"This candidate {phrases[0]}, {phrases[1]}, and {phrases[2]}."


def _is_incomplete_quality(quality: QualityScore) -> bool:
    """Return whether a rejected quality score primarily failed completeness."""

    if quality.incomplete_ending_penalty > 0:
        return True
    lowered_reasons = " ".join(quality.reasons).lower()
    return any(token in lowered_reasons for token in ("complete", "incomplete", "ending", "closure", "resolution"))


def _normalize_text(text: str) -> str:
    """Return punctuation-stripped lower-case text for duplicate checks."""

    return " ".join(TOKEN_RE.findall(text.lower()))


def _is_near_duplicate(left: str, right: str) -> bool:
    """Return whether two normalized strings are too similar to keep both."""

    if not left or not right:
        return False
    if left == right:
        return True

    ratio = SequenceMatcher(a=left, b=right).ratio()
    if ratio >= 0.9:
        return True

    left_tokens = left.split()
    right_tokens = right.split()
    if not left_tokens or not right_tokens:
        return False

    left_set = set(left_tokens)
    right_set = set(right_tokens)
    overlap = len(left_set & right_set) / max(len(left_set | right_set), 1)
    if overlap >= 0.78:
        return True

    leading_overlap = 0
    for left_token, right_token in zip(left_tokens, right_tokens):
        if left_token != right_token:
            break
        leading_overlap += 1
    return leading_overlap >= 4
