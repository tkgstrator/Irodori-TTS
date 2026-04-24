"""Shortcode-to-emoji mapping for Irodori-TTS style/effect annotations.

Upstream reference: https://huggingface.co/Aratako/Irodori-TTS-500M/blob/main/EMOJI_ANNOTATIONS.md
"""

from __future__ import annotations

import re

SHORTCODE_MAP: dict[str, str] = {
    "angry": "😠",
    "anxious": "😟",
    "backchannel": "👌",
    "cheerful": "😊",
    "chuckle": "🤭",
    "cough": "🤧",
    "cry": "😭",
    "drunk": "🥴",
    "echo": "📢",
    "exasperated": "🙄",
    "fast": "⏩",
    "gasp": "😮",
    "gentle": "🫶",
    "gulp": "🥤",
    "heavy_breath": "🌬️",
    "humming": "🎵",
    "joyful": "😆",
    "kiss": "💋",
    "lick": "👅",
    "muffled": "🤐",
    "painful": "😖",
    "panic": "😰",
    "pant": "🥵",
    "pause": "⏸️",
    "phone": "📞",
    "pleading": "🙏",
    "relieved": "😌",
    "scream": "😱",
    "shy": "🫣",
    "sigh": "😮‍💨",
    "sleepy": "😪",
    "slow": "🐢",
    "surprised": "😲",
    "teasing": "😏",
    "trembling": "🥺",
    "tsk": "😒",
    "whisper": "👂",
    "wondering": "🤔",
    "yawn": "🥱",
}

_SHORTCODE_RE = re.compile(r"\{([a-z_]+)\}")


def expand_shortcodes(text: str) -> str:
    """Replace ``{shortcode}`` occurrences with their emoji equivalents.

    Unknown shortcodes are left as-is.
    """
    return _SHORTCODE_RE.sub(
        lambda m: SHORTCODE_MAP.get(m.group(1), m.group(0)),
        text,
    )
