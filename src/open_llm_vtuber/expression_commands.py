"""
Chat commands for triggering Live2D expressions, persistent toggles, and
memory/affection management.

Commands are typed in the chat input with a `/` prefix (e.g. `/wave`, `/blush`).
They bypass the LLM and instantly trigger the corresponding action on the model.

Expression commands (face effects, poses) are transient — they play once via
the expression system. Toggle commands (body toggles) are persistent — they
directly set model parameters every frame via the injected frontend script.
The same toggle command typed again turns it off.

Context commands (/memories, /affection, /summarize, /forget) interact with
the memory and affection subsystems and require the ServiceContext.

The command map below is built for the witchZ3 model.
"""

import io
import json
import base64
from loguru import logger


def _generate_silent_wav(duration_ms: int = 100, sample_rate: int = 24000) -> str:
    """Generate a short silent WAV and return it as a base64 string.

    The frontend only applies expressions when audio data is present,
    so we send a tiny inaudible clip to carry the expression payload.
    """
    import struct

    num_samples = int(sample_rate * duration_ms / 1000)
    bits_per_sample = 16
    num_channels = 1
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(b"\x00" * data_size)

    return base64.b64encode(buf.getvalue()).decode("utf-8")


_SILENT_AUDIO = _generate_silent_wav()

# ── Transient expression commands (face effects & poses) ──
EXPRESSION_COMMANDS: dict[str, dict] = {
    # Face Effects (F-series, indices 0-10)
    "/star": {"index": 0, "label": "Star"},
    "/love": {"index": 1, "label": "Love"},
    "/flash": {"index": 2, "label": "Flash"},
    "/blush": {"index": 4, "label": "Blushing"},
    "/mask": {"index": 5, "label": "Mask"},
    "/tears": {"index": 6, "label": "Tears"},
    "/cry": {"index": 6, "label": "Tears"},
    "/exclaim": {"index": 7, "label": "!"},
    "/angry": {"index": 8, "label": "Angry"},
    "/sweat": {"index": 9, "label": "Sweat"},
    "/question": {"index": 10, "label": "?"},
    # Poses & Props (N-series, indices 11-21)
    "/lipstick": {"index": 11, "label": "Lipstick"},
    "/microphone": {"index": 12, "label": "Microphone"},
    "/scratch": {"index": 13, "label": "Cat Scratch"},
    "/whip": {"index": 14, "label": "Whip"},
    "/wave": {"index": 15, "label": "Waving"},
    "/cat": {"index": 16, "label": "Cat Paw"},
    "/hug": {"index": 17, "label": "Chest Hug"},
    "/hearts": {"index": 18, "label": "Heart Hands"},
    "/game": {"index": 19, "label": "Game"},
    "/write": {"index": 20, "label": "Writing"},
}

# ── Persistent toggle commands (body toggles via frontend injection) ──
TOGGLE_COMMANDS: dict[str, str] = {
    "/hideclothes": "Clothes (top + bottom)",
    "/hideunderwear": "Clothes + Underwear",
    "/hidebody": "Body",
    "/hidehead": "Head",
    "/shorthair": "Short Hair",
    "/hidecover": "Covering",
    "/hideearrings": "Earrings",
    "/hidex": "X covers",
    "/tongue": "Tongue Out",
    "/faceblack": "Face Black",
    "/yes": "Nod (yes)",
    "/no": "Shake (no)",
    "/tilt": "Tilt (curious)",
    "/confused": "Tilt (confused)",
    "/mic": "Mic Performance",
    "/dance": "Dance",
    "/excited": "Excited",
    "/wink": "Wink",
    "/think": "Thinking",
    "/shyaccept": "Shy Accept (test)",
    "/shyreject": "Shy Reject (test)",
    "/reset": "Reset All Toggles",
}

SENSITIVE_TOGGLES: dict[str, str] = {
    "/hideclothes": "Close",
    "/hideunderwear": "Intimate",
    "/hidex": "Intimate",
}

TIER_ORDER = ["Hostile", "Annoyed", "Stranger", "Acquaintance", "Friend", "Close", "Intimate"]

CONTEXT_COMMANDS = {"/memories", "/affection", "/summarize", "/forget"}

_ALL_COMMANDS = set(EXPRESSION_COMMANDS) | set(TOGGLE_COMMANDS) | CONTEXT_COMMANDS | {"/help"}


def is_expression_command(text: str) -> bool:
    """Return True if the text is a recognised /command."""
    token = text.strip().split()[0].lower() if text.strip() else ""
    return token in _ALL_COMMANDS


async def handle_expression_command(text: str, websocket_send, service_context=None) -> None:
    """Parse a /command and dispatch to expression, toggle, or context handler."""
    token = text.strip().split()[0].lower()

    if token == "/help":
        await _send_help(websocket_send)
        return

    if token in CONTEXT_COMMANDS:
        await _handle_context_command(token, websocket_send, service_context)
        return

    if token in TOGGLE_COMMANDS:
        await _handle_toggle(token, websocket_send, service_context)
        return

    cmd = EXPRESSION_COMMANDS.get(token)
    if not cmd:
        return

    logger.info(f"Expression command: {token} → index {cmd['index']} ({cmd['label']})")

    payload = {
        "type": "audio",
        "audio": _SILENT_AUDIO,
        "volumes": [0.0] * 5,
        "slice_length": 20,
        "display_text": {
            "text": f"✨ {cmd['label']}",
            "name": "Command",
            "avatar": None,
        },
        "actions": {"expressions": [cmd["index"]]},
        "forwarded": False,
    }
    await websocket_send(json.dumps(payload))


def _tier_reached(current_tier: str, required_tier: str) -> bool:
    """Return True if current_tier is at or above required_tier."""
    try:
        return TIER_ORDER.index(current_tier) >= TIER_ORDER.index(required_tier)
    except ValueError:
        return False


async def _handle_toggle(token: str, websocket_send, ctx=None) -> None:
    """Send a toggle-parameter message that the frontend script intercepts.

    For sensitive toggles, checks affection tier and plays a reaction sequence
    instead of (or alongside) the raw toggle.
    """
    label = TOGGLE_COMMANDS[token]
    required_tier = SENSITIVE_TOGGLES.get(token)

    if required_tier and ctx and hasattr(ctx, "affection") and ctx.affection:
        current_tier = ctx.affection.get_tier_name()
        unlocked = _tier_reached(current_tier, required_tier)

        if unlocked:
            logger.info(f"Toggle command: {token} ({label}) — affection OK ({current_tier}), shy-accept")
            await websocket_send(json.dumps({
                "type": "toggle-parameter",
                "command": token,
            }))
            await websocket_send(json.dumps({
                "type": "play-reaction",
                "reaction": "shy-accept",
            }))
            await websocket_send(json.dumps({
                "type": "full-text",
                "text": f"🔀 Toggle: {label}",
            }))
        else:
            logger.info(
                f"Toggle command: {token} ({label}) — affection too low "
                f"({current_tier}, needs {required_tier}), shy-reject"
            )
            await websocket_send(json.dumps({
                "type": "play-reaction",
                "reaction": "shy-reject",
            }))
            await websocket_send(json.dumps({
                "type": "full-text",
                "text": f"😤 Not happening. (need {required_tier} tier)",
            }))
        return

    logger.info(f"Toggle command: {token} ({label})")
    await websocket_send(json.dumps({
        "type": "toggle-parameter",
        "command": token,
    }))
    await websocket_send(json.dumps({
        "type": "full-text",
        "text": f"🔀 Toggle: {label}",
    }))


async def _handle_context_command(token: str, websocket_send, ctx) -> None:
    """Handle memory and affection inspection/management commands."""
    if ctx is None:
        await websocket_send(json.dumps({
            "type": "full-text",
            "text": "Memory system not available.",
        }))
        return

    if token == "/memories":
        ltm = ctx.long_term_memory
        if ltm and ltm.fact_count > 0:
            facts = ltm.get_facts()
            lines = [f"• {f}" for f in facts]
            text = f"Long-term memories ({len(facts)}):\n" + "\n".join(lines)
        else:
            text = "No long-term memories stored yet."
        await websocket_send(json.dumps({"type": "full-text", "text": text}))

    elif token == "/affection":
        aff = ctx.affection
        if aff:
            text = (
                f"Affection: {aff.level}/100 (tier: {aff.get_tier_name()})\n"
                f"Sessions: {aff.session_count}\n"
                f"Unlocked toggles: {', '.join(aff.get_unlocked_toggles()) or 'none'}"
            )
        else:
            text = "Affection system not available."
        await websocket_send(json.dumps({"type": "full-text", "text": text}))

    elif token == "/summarize":
        agent = ctx.agent_engine
        ltm = ctx.long_term_memory
        cm = ctx.context_manager
        if hasattr(agent, "get_memory") and hasattr(agent, "get_llm") and ltm and cm:
            memory = agent.get_memory()
            if len(memory) < 4:
                await websocket_send(json.dumps({
                    "type": "full-text",
                    "text": "Not enough conversation to summarize.",
                }))
                return

            try:
                from prompts import prompt_loader
                summary_prompt = prompt_loader.load_util("memory_summary_prompt")
            except Exception:
                summary_prompt = "Extract key facts from this conversation."

            facts = await cm.summarize_messages(memory, agent.get_llm(), summary_prompt)
            if facts:
                ltm.add_facts(facts)
                lines = [f"• {f}" for f in facts]
                text = f"Extracted {len(facts)} facts:\n" + "\n".join(lines)
            else:
                text = "No key facts extracted."
            await websocket_send(json.dumps({"type": "full-text", "text": text}))
        else:
            await websocket_send(json.dumps({
                "type": "full-text",
                "text": "Summarization not available.",
            }))

    elif token == "/forget":
        ltm = ctx.long_term_memory
        if ltm:
            count = ltm.fact_count
            ltm.clear()
            text = f"Cleared {count} long-term memories."
        else:
            text = "Memory system not available."
        await websocket_send(json.dumps({"type": "full-text", "text": text}))


async def _send_help(websocket_send) -> None:
    """Send a compact help listing of all available commands."""
    faces = "/star /love /flash /faceblack /blush /mask /tears /cry /exclaim /angry /sweat /question"
    poses = "/lipstick /scratch /whip /wave /cat /hug /hearts /game /write"
    toggles = "/hideclothes /hideunderwear /hidebody /hidehead /shorthair /hidecover /hideearrings /tongue /reset"
    anims = "/yes /no /tilt /confused /dance /mic /excited /wink /think"
    memory = "/memories /affection /summarize /forget"

    text = (
        f"Face: {faces}\n"
        f"Poses: {poses}\n"
        f"Toggle (persistent): {toggles}\n"
        f"Animations: {anims}\n"
        f"Memory: {memory}"
    )

    await websocket_send(
        json.dumps({"type": "full-text", "text": text})
    )
