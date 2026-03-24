"""
Parse affection-gated toggle tags from LLM responses.

When the AI outputs tags like [hideclothes], [hideunderwear], etc., this module
strips them from the response text and sends the corresponding toggle-parameter
messages to the frontend -- but only if the current affection tier allows it.

Also normalises hallucinated bracket tags (e.g. [nods] → [nod], [giggles] → [joy])
and strips any remaining unrecognised bracket tags so they are never spoken aloud.

When the model uses RP asterisks (*waves*, *giggles*) instead of brackets, those
segments are mapped to the same canonical [tag] vocabulary where possible; unknown
*...* segments are removed so they are not spoken.
"""

import re
import json
from loguru import logger


# ── Gesture / toggle tags → frontend commands ──────────────────────────
TAG_TO_COMMAND = {
    # Body motions (canonical) — each maps to a REACTION_DEFS entry in toggle-inject.js
    "nod": "/yes",
    "shake": "/no",
    "tilt": "/tilt",
    "confused": "/confused",
    "dance": "/dance",
    "excited": "/excited",
    "wink": "/wink",
    "think": "/think",
    "tongue": "/tongue",
    "mic": "/mic",
    "faceblack": "/faceblack",
    "shyaccept": "/shyaccept",
    "shyreject": "/shyreject",
    # Body motion aliases (common hallucinations)
    "nods": "/yes",
    "nodding": "/yes",
    "agree": "/yes",
    "shakes": "/no",
    "disagree": "/no",
    "tilts": "/tilt",
    "winks": "/wink",
    "dances": "/dance",
    "dancing": "/dance",
    "celebrate": "/dance",
    "thinks": "/think",
    "ponder": "/think",
    "pondering": "/think",
    "tongueout": "/tongue",
    "lick": "/tongue",
    "lipstick": "/tongue",
    "microphone": "/mic",
    "sing": "/mic",
    "singing": "/mic",
    "karaoke": "/mic",
    # Toggles (canonical)
    "hideclothes": "/hideclothes",
    "hidecover": "/hidecover",
    "hideunderwear": "/hideunderwear",
    "hidex": "/hidex",
    "reset": "/reset",
    # Toggle aliases
    "undress": "/hideclothes",
    "undresses": "/hideclothes",
    "strip": "/hideclothes",
    "strips": "/hideclothes",
    "disrobe": "/hideclothes",
    "robe": "/hideclothes",
    "nude": "/hidex",
    "naked": "/hidex",
    "fullnude": "/hidex",
    "dress": "/reset",
    "redress": "/reset",
}

ALWAYS_ALLOWED_TAGS = {
    "nod", "nods", "nodding", "agree",
    "shake", "shakes", "disagree",
    "tilt", "tilts",
    "confused",
    "dance", "dances", "dancing", "celebrate",
    "excited",
    "wink", "winks",
    "think", "thinks", "ponder", "pondering",
    "tongue", "tongueout", "lick", "lipstick",
    "mic", "microphone", "sing", "singing", "karaoke",
    "faceblack",
    "shyaccept", "shyreject",
    "reset",
    "dress", "redress",
}

TAG_PATTERN = re.compile(
    r"\[(" + "|".join(re.escape(k) for k in TAG_TO_COMMAND.keys()) + r")\]",
    re.IGNORECASE,
)

# Gesture + toggle bracket names (not in Live2D emotionMap) — must survive
# ``Live2dModel.strip_invalid_tags`` until ``extract_sentence_toggle_tags`` runs.
BRACKET_TAGS_GESTURE_AND_TOGGLE: frozenset[str] = frozenset(
    k.lower() for k in TAG_TO_COMMAND.keys()
)


def all_bracket_tags_for_display_strip(emo_map_keys: set[str]) -> set[str]:
    """Union of face-expression keys and motion/toggle keys for strip_unknown_tags."""
    return {k.lower() for k in emo_map_keys} | BRACKET_TAGS_GESTURE_AND_TOGGLE


BODY_MOTION_COMMANDS: frozenset[str] = frozenset({
    "/yes", "/no", "/tilt", "/confused", "/dance", "/excited",
    "/wink", "/think", "/tongue", "/mic", "/faceblack",
    "/shyaccept", "/shyreject",
})

CLOTHING_TOGGLE_COMMANDS: frozenset[str] = frozenset({
    "/hideclothes", "/hidecover", "/hideunderwear", "/hidex", "/reset",
})

SHY_REACT_TRIGGERS: frozenset[str] = frozenset({
    "/hideclothes", "/hideunderwear", "/hidex",
})

MAX_BODY_MOTIONS_PER_SENTENCE = 3

# ── Expression aliases (hallucinated → canonical expression tag) ───────
# These get rewritten in-place so the Live2D expression system picks them up.
EXPRESSION_ALIASES = {
    "giggles": "joy",
    "giggle": "joy",
    "laughs": "joy",
    "laugh": "joy",
    "laughing": "joy",
    "chuckles": "joy",
    "chuckle": "joy",
    "smiles": "joy",
    "smile": "joy",
    "grins": "smirk",
    "grin": "smirk",
    "smirks": "smirk",
    "smirk": "smirk",
    "sighs": "sadness",
    "sigh": "sadness",
    "pouts": "sadness",
    "pout": "sadness",
    "frowns": "sadness",
    "frown": "sadness",
    "gasps": "surprise",
    "gasp": "surprise",
    "scowls": "anger",
    "scowl": "anger",
    "shudders": "fear",
    "shudder": "fear",
    "trembles": "fear",
    "tremble": "fear",
    "blushes": "blush",
    "blushing": "blush",
    "cries": "cry",
    "crying": "cry",
    "wince": "sweat",
    "winces": "sweat",
    "gulps": "sweat",
    "gulp": "sweat",
}

_EXPR_ALIAS_PATTERN = re.compile(
    r"\[(" + "|".join(re.escape(k) for k in EXPRESSION_ALIASES.keys()) + r")\]",
    re.IGNORECASE,
)

# witchZ3 emotionMap keys — *joy* / *wave* etc. pass through to [joy] / [wave]
WITCH_EMOTION_KEYS = frozenset(
    {
        "star",
        "joy",
        "happy",
        "excited",
        "love",
        "gentle",
        "warm",
        "relieved",
        "smile",
        "smiles",
        "laugh",
        "laughs",
        "laughing",
        "giggle",
        "giggles",
        "chuckle",
        "chuckles",
        "cheerful",
        "delighted",
        "disgust",
        "grossed",
        "repulsed",
        "blush",
        "shy",
        "embarrassed",
        "smirk",
        "wry",
        "amused",
        "sarcastic",
        "proud",
        "confident",
        "smug",
        "teasing",
        "flustered",
        "blushing",
        "grin",
        "grins",
        "sadness",
        "sad",
        "disappointed",
        "sorry",
        "cry",
        "cries",
        "crying",
        "sigh",
        "sighs",
        "pout",
        "pouts",
        "frown",
        "frowns",
        "melancholy",
        "upset",
        "surprise",
        "surprised",
        "impressed",
        "exclaim",
        "gasp",
        "gasps",
        "shocked",
        "stunned",
        "anger",
        "angry",
        "annoyed",
        "frustrated",
        "determined",
        "serious",
        "scowl",
        "scowls",
        "furious",
        "mad",
        "irritated",
        "sweat",
        "nervous",
        "worried",
        "fear",
        "anxious",
        "scared",
        "frightened",
        "terrified",
        "shudder",
        "shudders",
        "tremble",
        "trembles",
        "gulp",
        "gulps",
        "wince",
        "winces",
        "uneasy",
        "question",
        "curious",
        "confused",
        "thinking",
        "puzzled",
        "skeptical",
        "doubtful",
        "ponder",
        "pondering",
        "scratch",
        "mischievous",
        "wave",
        "greeting",
        "cat",
        "playful",
        "hug",
        "comforting",
        "hearts",
        "flirty",
        "affection",
        "kiss",
        "adoring",
        "game",
        "write",
        "tongue",
        "cheeky",
    }
)

# Extra RP verbs → bracket tag (TAG_TO_COMMAND uses "wave" but not "waves")
_ASTERISK_VERBAL_EXTRA = {
    "waves": "wave",
    "waving": "wave",
    "tilting": "tilt",
    "leans": "tilt",
    "leaning": "tilt",
}


def _build_asterisk_inner_to_canonical() -> dict[str, str]:
    """Single-token *inner* (lowercase) → canonical tag for [...]."""
    m: dict[str, str] = {}
    for alias, canon in EXPRESSION_ALIASES.items():
        m[alias] = canon
    for k, v in _ASTERISK_VERBAL_EXTRA.items():
        m[k] = v
    for key in TAG_TO_COMMAND.keys():
        m[key] = key
    for ek in WITCH_EMOTION_KEYS:
        m.setdefault(ek, ek)
    return m


ASTERISK_INNER_TO_CANONICAL: dict[str, str] = _build_asterisk_inner_to_canonical()

# Words inside a longer *phrase* that imply [question] if no gesture matched first
_ASTERISK_CURIOSITY_TOKENS = frozenset(
    {
        "curiosity",
        "curious",
        "curiously",
        "wondering",
        "wonders",
        "puzzled",
    }
)

# Single *...* segments; avoids matching markdown **bold**
_ASTERISK_PAIR = re.compile(r"(?<!\*)\*([^*]+?)\*(?!\*)")


def _resolve_asterisk_inner(inner: str) -> str | None:
    """Map *inner* text to one canonical tag name, or None.

    Single token: direct lookup. Multi-word RP (*tilts head slightly in curiosity*):
    scan tokens left-to-right — first match wins (so *tilts* before *curiosity*).
    If nothing matches but a curiosity token appears, use [question].
    """
    inner = inner.strip().lower()
    inner = re.sub(r"\s+", " ", inner)
    if not inner:
        return None

    if " " not in inner:
        hit = ASTERISK_INNER_TO_CANONICAL.get(inner)
        if hit:
            return hit
        if inner in _ASTERISK_CURIOSITY_TOKENS:
            return "question"
        return None

    tokens = inner.split()
    for t in tokens:
        hit = ASTERISK_INNER_TO_CANONICAL.get(t)
        if hit:
            return hit
    for t in tokens:
        if t in _ASTERISK_CURIOSITY_TOKENS:
            return "question"
    return None


def map_asterisks_to_bracket_tags(text: str) -> str:
    """Turn RP asterisk actions into [tag] when we recognise the word.

    Multi-word segments scan for the first known verb/emotion token. Unmatched
    *...* segments are removed (not spoken). Bracket tags are left unchanged.
    """
    if "*" not in text:
        return text

    def _repl(match: re.Match) -> str:
        raw = match.group(1).strip()
        if not raw:
            return ""
        canon = _resolve_asterisk_inner(raw)
        if canon:
            logger.info(f"🔄 Asterisk RP mapped: *{raw}* → [{canon}]")
            return f"[{canon}]"
        logger.info(f"🗑️ Asterisk RP dropped (unknown): *{raw}*")
        return ""

    out = _ASTERISK_PAIR.sub(_repl, text)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()

# Catch-all: any remaining [bracketed content] not consumed by the
# expression system or the toggle/gesture parser.  Matches single-word
# tags like [dance] as well as multi-word hallucinations like
# [Think for a moment, then smile widely and nod].
_BRACKET_TAG = re.compile(r"\[[^\[\]]+\]")


# Strip meta-commentary that weaker models inject about the tag/animation system.
# Matches parenthetical asides like "(Note: The actual animations would not be …)"
_META_COMMENTARY = re.compile(
    r"\((?:Note|Disclaimer|Reminder|FYI|Info|Aside|Explanation|Context|Hint)"
    r"[^)]{10,}\)",
    re.IGNORECASE,
)


def cleanup_after_tag_strip(text: str) -> str:
    """Fix punctuation holes left when [bracket] tokens are removed from a list.

    Also strips parenthetical meta-commentary that local models hallucinate
    about the animation / tag system.

    Removing ``[hideclothes], [hidecover], and [reset]`` yields
    ``involve , , and ?`` — readable text must be repaired.
    """
    if not text:
        return text
    meta_found = _META_COMMENTARY.findall(text)
    if meta_found:
        logger.info(f"🗑️ Stripped meta-commentary: {meta_found}")
    text = _META_COMMENTARY.sub("", text)
    t = text
    # Common: "... involve , , and ?" -> "... involve?"
    t = re.sub(r",\s*,\s*and\s*\?", "?", t)
    t = re.sub(r",\s*,\s*and\s+", " ", t)
    for _ in range(6):
        t2 = re.sub(r",\s*,", ",", t)
        if t2 == t:
            break
        t = t2
    t = re.sub(r",\s*\?", "?", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def normalize_tags(text: str) -> str:
    """Replace hallucinated bracket tags with their canonical equivalents.

    Handles both expression aliases ([giggles] → [joy]) and
    gesture/toggle aliases ([nods] → [nod]) so downstream systems
    can recognise them.

    Also maps RP asterisks (*waves* → [wave]) before bracket alias fixes.
    """
    text = map_asterisks_to_bracket_tags(text)

    def _replace_expr(m):
        canonical = EXPRESSION_ALIASES.get(m.group(1).lower())
        if canonical:
            logger.info(f"🔄 Tag alias rewritten: [{m.group(1)}] → [{canonical}]")
        return f"[{canonical}]" if canonical else m.group(0)

    return _EXPR_ALIAS_PATTERN.sub(_replace_expr, text)


def strip_unknown_tags(text: str, known_tags: set[str] | None = None) -> str:
    """Remove any [bracket tags] that aren't in the known set.

    Call this AFTER expression extraction and toggle/gesture extraction
    to silently drop hallucinated tags that didn't match anything.

    Parameters:
        text:       The display text (expression tags already consumed).
        known_tags: Optional set of lowercase tag names to preserve.
                    If None, strips ALL remaining bracket tags.
    """
    stripped_tags: list[str] = []

    if known_tags is None:
        def _log_strip(m):
            stripped_tags.append(m.group(0))
            return ""
        cleaned = _BRACKET_TAG.sub(_log_strip, text).strip()
        if stripped_tags:
            logger.info(f"🗑️ strip_unknown_tags (no allowlist) removed: {stripped_tags}")
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleanup_after_tag_strip(cleaned)

    lower_known = {k.lower() for k in known_tags}

    def _keep_known(m):
        content = m.group(0)[1:-1].lower()
        if content in lower_known:
            return m.group(0)
        stripped_tags.append(m.group(0))
        return ""

    cleaned = _BRACKET_TAG.sub(_keep_known, text)
    if stripped_tags:
        logger.info(f"🗑️ strip_unknown_tags removed: {stripped_tags}")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleanup_after_tag_strip(cleaned)


def extract_sentence_toggle_tags(
    text: str,
    affection_tracker=None,
) -> tuple:
    """Extract toggle tags from a single sentence.

    Returns (cleaned_text, list_of_commands) so the caller can attach
    commands to the per-sentence Actions and strip tags from display text.
    """
    matches = TAG_PATTERN.findall(text)
    if not matches:
        return text, []

    unlocked = (
        set(affection_tracker.get_unlocked_toggles())
        if affection_tracker
        else set()
    )

    commands = []
    for tag_name in matches:
        tag_lower = tag_name.lower()
        command = TAG_TO_COMMAND.get(tag_lower)
        if not command:
            continue
        if tag_lower in ALWAYS_ALLOWED_TAGS or tag_lower in unlocked:
            logger.info(f"LLM tag extracted per-sentence: [{tag_name}] -> {command}")
            commands.append(command)
        else:
            logger.debug(
                f"Toggle [{tag_name}] not unlocked at current tier "
                f"(unlocked: {unlocked})"
            )

    cleaned = TAG_PATTERN.sub("", text).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleanup_after_tag_strip(cleaned)
    return cleaned, commands


async def parse_and_send_toggle_tags(
    response_text: str,
    websocket_send,
    affection_tracker,
) -> str:
    """Scan response for toggle tags, execute allowed ones, strip all from text.

    Returns the cleaned response text.
    """
    matches = TAG_PATTERN.findall(response_text)
    if not matches:
        return response_text

    unlocked = set(affection_tracker.get_unlocked_toggles())

    for tag_name in matches:
        tag_lower = tag_name.lower()
        command = TAG_TO_COMMAND.get(tag_lower)
        if not command:
            continue

        if tag_lower in ALWAYS_ALLOWED_TAGS or tag_lower in unlocked:
            logger.info(f"LLM tag fired: [{tag_name}] -> {command}")
            msg = {
                "type": "toggle-parameter",
                "command": command,
            }
            if command != "/reset":
                msg["force"] = "on"
            await websocket_send(json.dumps(msg))
        else:
            logger.debug(
                f"Toggle [{tag_name}] not unlocked at current tier "
                f"(unlocked: {unlocked})"
            )

    cleaned = TAG_PATTERN.sub("", response_text).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleanup_after_tag_strip(cleaned)
    return cleaned
