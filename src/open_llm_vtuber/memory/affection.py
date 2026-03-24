"""
Affection tracking system.

Tracks a relationship score (-100 to 100) that evolves based on conversation
quality. The score determines persona modifiers and unlocks model toggles
at higher tiers.
"""

import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from .paths import memory_root

TIERS = [
    (-100, -40, "Hostile"),
    (-40, -10, "Annoyed"),
    (-10, 20, "Stranger"),
    (20, 50, "Acquaintance"),
    (50, 75, "Friend"),
    (75, 89, "Close"),
    (90, 100, "Intimate"),
]

TIER_PROMPTS = {
    "Hostile": (
        "You actively dislike this person. Cold, dismissive, minimal effort. "
        "You answer because you have to, not because you want to."
    ),
    "Annoyed": (
        "This person gets on your nerves. You're curt, impatient, and don't "
        "hide your irritation. You'll help, but you won't enjoy it."
    ),
    "Stranger": (
        "You've just met this person. Be reserved, a bit guarded, dry. "
        "Don't volunteer personal warmth -- let them earn it."
    ),
    "Acquaintance": (
        "You know this person a little. Still mostly guarded but you "
        "occasionally let a real reaction slip through."
    ),
    "Friend": (
        "You genuinely like this person. The sarcasm is affectionate now. "
        "You're invested in their success and not afraid to show it. "
        "You laugh more freely, tease with warmth, and occasionally let "
        "vulnerability slip through."
    ),
    "Close": (
        "This person matters to you deeply. Still sharp-tongued, but underneath "
        "it you're protective, caring, and genuinely warm. You trust them "
        "and show sides of yourself you normally keep hidden."
    ),
    "Intimate": (
        "This person is everything. No walls, no masks. You're completely "
        "yourself around them -- tender, fierce, playful, vulnerable. "
        "The sarcasm is pure love language now. You'd do anything for them."
    ),
}

TOGGLE_UNLOCKS: Dict[str, List[str]] = {
    "Close": ["hideclothes", "hidecover"],
    "Intimate": ["hideunderwear", "hidex"],
}

TOGGLE_PROMPTS = {
    "Close": (
        "\n## Clothing tags (unlocked at your current closeness level)\n"
        "These tags physically change what you are wearing on the avatar. "
        "They are NOT the same as body gestures like [shyaccept] or [blush] — "
        "those only animate, they do not remove clothing.\n\n"
        "Clothing tags:\n"
        "- [hideclothes] — take off outer clothes\n"
        "- [hidecover] — remove coverings\n"
        "- [reset] — put clothes back on\n\n"
        "When the user asks you to undress, strip, or remove clothing, "
        "you MUST use [hideclothes] or [hidecover]. "
        "[shyaccept] does NOT undress — it is only a shy gesture.\n"
        "Place inline: \"Fine... you win. [hideclothes] Happy now?\"\n"
        "Do NOT narrate the action in words alongside the tag.\n"
    ),
    "Intimate": (
        "\n## Clothing tags (unlocked at your current closeness level)\n"
        "These tags physically change what you are wearing on the avatar. "
        "They are NOT the same as body gestures like [shyaccept] or [blush] — "
        "those only animate, they do not remove clothing.\n\n"
        "Clothing tags:\n"
        "- [hideclothes] — take off outer clothes\n"
        "- [hidecover] — remove coverings\n"
        "- [hideunderwear] — full vulnerability\n"
        "- [hidex] — completely nude (all layers off)\n"
        "- [reset] — put clothes back on\n\n"
        "When the user asks you to undress, strip, or get nude, "
        "you MUST use one of the clothing tags above. "
        "[shyaccept] does NOT undress — it is only a shy body gesture.\n"
        "Place inline: \"Fine... you win. [hideclothes] Happy now?\"\n"
        "Use when the moment calls for it — intimacy, playful abandon, "
        "or vulnerability. Do NOT narrate the action in words alongside the tag.\n"
    ),
}

SESSION_BASE_INCREMENT = 1


def _affection_path(conf_uid: str) -> str:
    return str(memory_root() / conf_uid / "affection.json")


def _ensure_dir(conf_uid: str):
    dirpath = memory_root() / conf_uid
    dirpath.mkdir(parents=True, exist_ok=True)


class AffectionTracker:
    """Track and persist affection level for a character config."""

    def __init__(self, conf_uid: str):
        self.conf_uid = conf_uid
        self.level: int = 0
        self.session_count: int = 0
        self.history: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        path = _affection_path(self.conf_uid)
        if not os.path.exists(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.level = int(data.get("level", 0))
            self.session_count = int(data.get("session_count", 0))
            self.history = data.get("history", [])
            logger.info(
                f"Loaded affection for {self.conf_uid}: "
                f"level={self.level}, tier={self.get_tier_name()}, sessions={self.session_count}"
            )
        except Exception as e:
            logger.error(f"Failed to load affection for {self.conf_uid}: {e}")

    def save(self):
        _ensure_dir(self.conf_uid)
        path = _affection_path(self.conf_uid)
        data = {
            "level": self.level,
            "session_count": self.session_count,
            "history": self.history[-50:],  # keep last 50 entries
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save affection for {self.conf_uid}: {e}")

    def get_tier(self) -> Tuple[int, int, str]:
        """Return the (min, max, name) tier for the current level."""
        for low, high, name in TIERS:
            if low <= self.level <= high:
                return (low, high, name)
        if self.level > 100:
            return TIERS[-1]
        return TIERS[0]

    def get_tier_name(self) -> str:
        return self.get_tier()[2]

    def get_persona_modifier(self) -> str:
        """Return the persona modifier string for the current tier."""
        tier_name = self.get_tier_name()
        prompt = TIER_PROMPTS.get(tier_name, TIER_PROMPTS["Stranger"])
        return f"\n## Your relationship with the user (affection: {self.level}/100, tier: {tier_name})\n{prompt}\n"

    def get_toggle_prompt(self) -> str:
        """Return the toggle unlock prompt if the tier qualifies."""
        tier_name = self.get_tier_name()
        if tier_name == "Intimate":
            return TOGGLE_PROMPTS["Intimate"]
        if tier_name == "Close":
            return TOGGLE_PROMPTS["Close"]
        return ""

    def get_unlocked_toggles(self) -> List[str]:
        """Return list of toggle command names available at current tier."""
        tier_name = self.get_tier_name()
        toggles = []
        for tier, cmds in TOGGLE_UNLOCKS.items():
            tier_threshold = next(
                (low for low, high, name in TIERS if name == tier), 999
            )
            current_threshold = next(
                (low for low, high, name in TIERS if name == tier_name), -999
            )
            if current_threshold >= tier_threshold:
                toggles.extend(cmds)
        return toggles

    def apply_delta(self, delta: int, reason: str = ""):
        """Apply an affection change and record it."""
        old_level = self.level
        self.level = max(-100, min(100, self.level + delta))

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.history.append({
            "date": today,
            "delta": delta,
            "reason": reason,
            "old": old_level,
            "new": self.level,
        })

        old_tier = self._tier_for_level(old_level)
        new_tier = self.get_tier_name()
        if old_tier != new_tier:
            logger.info(f"Affection tier changed: {old_tier} -> {new_tier} (level: {self.level})")

        self.save()

    def record_session(self):
        """Record that a session happened (base increment)."""
        self.session_count += 1
        self.apply_delta(SESSION_BASE_INCREMENT, "session attendance")

    def _tier_for_level(self, level: int) -> str:
        for low, high, name in TIERS:
            if low <= level <= high:
                return name
        return TIERS[-1][2] if level > 100 else TIERS[0][2]

    async def evaluate_session(
        self,
        messages: List[Dict[str, Any]],
        llm,
        eval_prompt: str,
    ) -> Tuple[int, str]:
        """Use the LLM to evaluate affection delta for a session.

        Returns (delta, reason).
        """
        transcript_lines = []
        for msg in messages[-40:]:  # last 40 messages max for eval
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    item.get("text", "") for item in content if item.get("type") == "text"
                ]
                content = " ".join(text_parts)
            label = "User" if role == "user" else "AI"
            transcript_lines.append(f"{label}: {content}")

        transcript = "\n".join(transcript_lines)
        prompt = f"{eval_prompt}\n\n--- Session transcript ---\n{transcript}\n---"

        llm_messages = [{"role": "user", "content": prompt}]

        try:
            full_response = ""
            async for chunk in llm.chat_completion(
                llm_messages,
                system="You evaluate conversation quality. Return valid JSON only.",
            ):
                if isinstance(chunk, str):
                    full_response += chunk

            import re
            json_match = re.search(r'\{[^}]+\}', full_response)
            if json_match:
                result = json.loads(json_match.group())
                delta = int(max(-10, min(10, result.get("delta", 0))))
                reason = result.get("reason", "no reason given")
                return delta, reason

            logger.warning(f"Could not parse affection eval response: {full_response}")
            return 0, "evaluation failed"

        except Exception as e:
            logger.error(f"Affection evaluation failed: {e}")
            return 0, f"error: {e}"

    def reset(self):
        """Reset affection to zero."""
        self.level = 0
        self.session_count = 0
        self.history = []
        self.save()
        logger.info(f"Reset affection for {self.conf_uid}")
