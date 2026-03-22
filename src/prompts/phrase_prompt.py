"""Format-first prompt builders for generation requests."""

from __future__ import annotations

from app.schemas import CreativeBrief, GenerateSingleRequest, OutputSpec

_VOICE_PACK_GUIDANCE = {
    "romantic_witty": [
        "Use flirtatious warmth with quick, memorable phrasing.",
        "Keep the humor charming, not sarcastic or crude.",
        "Avoid overblown declarations unless explicitly requested.",
    ],
    "festival_warm": [
        "Keep the tone celebratory, warm, and family-friendly.",
        "Use occasion details naturally without sounding templated.",
        "Prefer light, image-rich phrasing over generic blessing language.",
    ],
    "festival_respectful": [
        "Keep the tone respectful, gracious, and spiritually aware.",
        "Avoid slang, parody, or flippant humor.",
        "Use calm warmth and sincere goodwill.",
    ],
    "playful_modern": [
        "Sound current, conversational, and lightly witty.",
        "Use pop-culture cadence only when it feels natural.",
        "Keep the copy crisp and avoid trying too hard to be trendy.",
    ],
    "minimal_heartfelt": [
        "Keep the language clean, soft, and emotionally direct.",
        "Prefer understated sincerity over decorative wording.",
        "Let the emotional line land without extra explanation.",
    ],
}

_ROMANTIC_HINTS = ("valentine", "romance", "romantic", "love", "crush", "date", "anniversary")
_PLAYFUL_HINTS = ("netflix", "chill", "sofa", "couch", "quirky", "meme", "fun", "banter")
_RESPECTFUL_FESTIVAL_HINTS = ("ramadan", "ramazan", "eid", "iftar", "dua", "blessing")
_WARM_FESTIVAL_HINTS = ("diwali", "holi", "christmas", "pongal", "onam", "festival", "lights", "rangoli")


def tone_direction(tone_funny_pct: int, tone_emotion_pct: int) -> str:
    """Map tone sliders to a concise style direction."""

    if tone_funny_pct >= 70:
        return "Strongly witty and playful, while still respectful."
    if tone_emotion_pct >= 70:
        return "Strongly heartfelt and emotionally resonant."
    if tone_funny_pct >= 55:
        return "Light humor with balanced sincerity."
    if tone_emotion_pct >= 55:
        return "Warm and emotional with restrained humor."
    return "Balanced and clean conversational tone."


def cultural_context_guidance(cultural_context: str) -> str:
    """Return culturally-aware guidance for prompt conditioning."""

    guidance = {
        "global": "Use globally neutral phrasing unless localized context is explicitly requested.",
        "indian": "When relevant, favor grounded warmth with familiar India-friendly social context.",
        "bengali": "When relevant, use softer reflective tone with adda/chai/rain/Kolkata-friendly imagery.",
        "punjabi": "When relevant, use warm expressive family/community energy and celebration references.",
        "south_indian": "When relevant, use grounded warmth with home/ritual/food/cultural familiarity.",
        "western": "Use neutral western phrasing and cadence.",
        "american": "Use casual direct phrasing with familiar pop-cultural cadence when relevant.",
        "asian": "When relevant, keep restraint and subtle emotionality.",
    }
    return guidance.get(cultural_context, guidance["global"])


def selected_voice_pack(payload: GenerateSingleRequest) -> str:
    """Return the resolved voice pack for one request."""

    brief = payload.creative_brief
    if brief is not None and brief.voice_pack != "auto":
        return brief.voice_pack

    haystack = " ".join(
        [
            payload.theme_name,
            payload.visual_style,
            " ".join(payload.prompt_keywords),
        ]
    ).lower()

    if any(token in haystack for token in _RESPECTFUL_FESTIVAL_HINTS):
        return "festival_respectful"
    if any(token in haystack for token in _ROMANTIC_HINTS):
        return "romantic_witty" if payload.tone_funny_pct >= 35 else "minimal_heartfelt"
    if any(token in haystack for token in _PLAYFUL_HINTS):
        return "playful_modern"
    if any(token in haystack for token in _WARM_FESTIVAL_HINTS):
        return "festival_warm"
    if payload.tone_funny_pct >= 60:
        return "playful_modern"
    if payload.tone_emotion_pct >= 65:
        return "minimal_heartfelt"
    return "festival_warm" if payload.app_id == "ecard_factory" else "minimal_heartfelt"


def _format_tone_blend(brief: CreativeBrief | None) -> str | None:
    """Render tone blend into one concise guideline line."""

    if brief is None or not brief.tone_blend:
        return None
    parts = [f"{tone} {score}%" for tone, score in sorted(brief.tone_blend.items())]
    return ", ".join(parts)


def creative_brief_guidance(payload: GenerateSingleRequest) -> list[str]:
    """Return creative-brief-specific guidance lines."""

    brief = payload.creative_brief
    voice_pack = selected_voice_pack(payload)
    lines = [f"- Voice pack: {voice_pack}."]

    for instruction in _VOICE_PACK_GUIDANCE.get(voice_pack, []):
        lines.append(f"- {instruction}")

    tone_blend = _format_tone_blend(brief)
    if tone_blend:
        lines.append(f"- Tone blend target: {tone_blend}.")

    if brief is None:
        return lines

    if brief.audience_age_band:
        lines.append(f"- Audience age band: {brief.audience_age_band}.")
    if brief.cultural_guardrails:
        lines.append(f"- Cultural guardrails: {', '.join(brief.cultural_guardrails)}.")
    if brief.taboo_phrases:
        lines.append(f"- Extra taboo phrases: {', '.join(brief.taboo_phrases)}.")
    if brief.target_structure:
        lines.append(f"- Creative target structure: {brief.target_structure}.")
    if brief.desired_emotional_effect:
        lines.append(f"- Desired emotional effect: {brief.desired_emotional_effect}.")
    return lines


def format_template(spec: OutputSpec) -> str:
    """Return the explicit template block for one output format."""

    if spec.format == "one_liner":
        lines = spec.structure.items or 3
        numbering_allowed = not bool(spec.structure.no_numbering)
        numbering_rule = "numbering allowed" if numbering_allowed else "no numbering"
        return (
            f"Template: one_liner\n"
            f"- Return exactly {lines} lines.\n"
            f"- Use one line per item; {numbering_rule}.\n"
            "- No JSON."
        )

    if spec.format == "paragraph":
        return (
            "Template: paragraph\n"
            "- Return a single paragraph.\n"
            "- Use 3 to 6 sentences.\n"
            "- Plain text only."
        )

    if spec.format == "one_page":
        return (
            "Template: one_page\n"
            "- Return 2 to 4 short paragraphs.\n"
            "- Headings are optional.\n"
            "- Plain text only."
        )

    if spec.format == "pros_cons":
        items_per_section = spec.structure.items or 4
        return (
            "Template: pros_cons\n"
            '- Return exactly two sections: "Pros:" and "Cons:".\n'
            f"- Use exactly {items_per_section} bullet points per section.\n"
            "- No extra text before or after sections."
        )

    if spec.format == "verse":
        min_lines = spec.structure.items or 8
        max_lines = spec.structure.max_lines or 12
        return (
            "Template: verse\n"
            f"- Return {min_lines} to {max_lines} lines only.\n"
            "- No paragraph blocks.\n"
            "- No title unless explicitly requested."
        )

    return (
        "Template: story\n"
        "- Return a short story in exactly 3 sections.\n"
        "- Section headers must be: Setup, Turn, Resolution.\n"
        "- Plain text only."
    )


def length_constraints(spec: OutputSpec) -> list[str]:
    """Return normalized length constraints as human-readable lines."""

    lines: list[str] = []
    if spec.length.min_words is not None:
        lines.append(f"- Minimum words: {spec.length.min_words}.")
    if spec.length.max_words is not None:
        lines.append(f"- Maximum words: {spec.length.max_words}.")
    if spec.length.target_words is not None:
        lines.append(f"- Target words: {spec.length.target_words}.")
    if spec.structure.max_words_per_line is not None:
        lines.append(f"- Max words per line: {spec.structure.max_words_per_line}.")
    return lines


def build_system_prompt() -> str:
    """Return global hard constraints for all formats and backends."""

    return (
        "You are a production copywriting assistant.\n"
        "Hard rules:\n"
        "- Return only requested output content.\n"
        "- Never return JSON, markdown, XML, labels, or explanations.\n"
        "- Never prefix output with 'Sure' or 'Here's'.\n"
        "- Keep wording natural and readable.\n"
        "- Respect all format and policy constraints exactly."
    )


def build_guidelines_prompt(payload: GenerateSingleRequest) -> str:
    """Return format-first policy and style guidance."""

    spec = payload.output_spec or OutputSpec()
    emoji_instruction = {
        "none": "Do not use emojis.",
        "light": "Use at most one subtle emoji where natural.",
        "expressive": "Emojis are allowed where natural.",
    }[payload.emoji_policy]

    avoid_instruction = "Avoid common cliches."
    if payload.avoid_cliches:
        banned = ", ".join(payload.avoid_phrases) if payload.avoid_phrases else "none provided"
        avoid_instruction = f"Avoid cliches and avoid these exact phrases: {banned}."
    if payload.creative_brief is not None and payload.creative_brief.taboo_phrases:
        extra_banned = ", ".join(payload.creative_brief.taboo_phrases)
        avoid_instruction = f"{avoid_instruction} Also avoid these taboo phrases: {extra_banned}."

    length_lines = "\n".join(length_constraints(spec)) or "- No explicit length override."
    brief_lines = "\n".join(creative_brief_guidance(payload))

    return (
        "GUIDELINES\n"
        f"- App ID: {payload.app_id}\n"
        f"- Content type: {payload.content_type}\n"
        f"- Tone direction: {tone_direction(payload.tone_funny_pct, payload.tone_emotion_pct)}\n"
        f"- Tone style: {payload.tone_style}\n"
        f"- Audience: {payload.audience}\n"
        f"- Cultural context: {payload.cultural_context}\n"
        f"- Cultural guidance: {cultural_context_guidance(payload.cultural_context)}\n"
        "- Do not stereotype or force cultural markers.\n"
        "- Use cultural context only when relevant to the request.\n"
        "- Keep cultural references natural.\n"
        f"- Emoji policy: {emoji_instruction}\n"
        f"- {avoid_instruction}\n"
        "- Creative brief:\n"
        f"{brief_lines}\n"
        "- Must not return JSON.\n"
        "- Must not prefix with 'Sure' or 'Here's'.\n"
        "- Format template:\n"
        f"{format_template(spec)}\n"
        "- Length constraints:\n"
        f"{length_lines}"
    )


def build_user_prompt(payload: GenerateSingleRequest) -> str:
    """Return task-level request data."""

    keywords = ", ".join(payload.prompt_keywords) if payload.prompt_keywords else "none"
    return (
        "USER TASK\n"
        f"Theme: {payload.theme_name}\n"
        f"Visual style: {payload.visual_style}\n"
        f"Keywords to include naturally when helpful: {keywords}\n"
        "Prioritize originality, emotional believability, and clean completion.\n"
        "Output plain text only."
    )


def build_messages(payload: GenerateSingleRequest) -> list[dict[str, str]]:
    """Return SYSTEM + USER messages with all generation constraints."""

    user_content = "\n\n".join(
        [
            build_guidelines_prompt(payload),
            build_user_prompt(payload),
        ]
    )
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_content},
    ]
