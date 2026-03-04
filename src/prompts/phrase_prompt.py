"""Phrase generation prompt builders."""

from __future__ import annotations

from app.schemas import GenerateSingleRequest


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


def build_system_prompt() -> str:
    """Return non-negotiable generation constraints."""

    return (
        "You are a production copywriting assistant for short greeting phrases.\n"
        "Hard rules:\n"
        "- Return only the requested phrases and nothing else.\n"
        "- Never return JSON, markdown, XML, code blocks, labels, or explanations.\n"
        "- Keep each phrase concise, natural, and human-sounding.\n"
        "- Respect user constraints for count, word limits, emoji policy, and style."
    )


def build_guidelines_prompt(payload: GenerateSingleRequest) -> str:
    """Return policy and style guidance derived from request options."""

    emoji_instruction = {
        "none": "Do not use emojis.",
        "light": "Use at most one subtle emoji per phrase when it helps tone.",
        "expressive": "Emojis are allowed when natural, but keep readability first.",
    }[payload.emoji_policy]

    format_instruction = {
        "lines": "Output as plain lines without numbering.",
        "numbered": "Output as a numbered list with one phrase per line.",
    }[payload.output_format]

    avoid_fragment = "Avoid common cliches."
    if payload.avoid_cliches:
        banned = ", ".join(payload.avoid_phrases) if payload.avoid_phrases else "none provided"
        avoid_fragment = f"Avoid cliches and avoid these phrases exactly: {banned}."

    return (
        "GUIDELINES\n"
        f"- Tone direction: {tone_direction(payload.tone_funny_pct, payload.tone_emotion_pct)}\n"
        f"- Tone style: {payload.tone_style}\n"
        f"- Audience: {payload.audience}\n"
        f"- Maximum words per phrase: {payload.max_words}\n"
        f"- Emoji policy: {emoji_instruction}\n"
        f"- {avoid_fragment}\n"
        f"- {format_instruction}\n"
        "- Do not repeat near-duplicate phrases."
    )


def build_user_prompt(payload: GenerateSingleRequest) -> str:
    """Return the concrete phrase-generation task."""

    keywords = ", ".join(payload.prompt_keywords) if payload.prompt_keywords else "none"
    return (
        "USER TASK\n"
        f"Theme: {payload.theme_name}\n"
        f"Visual style: {payload.visual_style}\n"
        f"Keywords to include naturally when helpful: {keywords}\n"
        f"Generate exactly {payload.count} phrases.\n"
        "Each phrase must be a standalone phrase suitable for social sharing."
    )


def build_messages(payload: GenerateSingleRequest) -> list[dict[str, str]]:
    """Return a 3-layer prompt sequence (SYSTEM + GUIDELINES + USER TASK)."""

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
