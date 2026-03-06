"""Format-first prompt builders for generation requests."""

from __future__ import annotations

from app.schemas import GenerateSingleRequest, OutputSpec


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

    length_lines = "\n".join(length_constraints(spec)) or "- No explicit length override."

    return (
        "GUIDELINES\n"
        f"- Tone direction: {tone_direction(payload.tone_funny_pct, payload.tone_emotion_pct)}\n"
        f"- Tone style: {payload.tone_style}\n"
        f"- Audience: {payload.audience}\n"
        f"- Emoji policy: {emoji_instruction}\n"
        f"- {avoid_instruction}\n"
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
