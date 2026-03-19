"""
Persistent key-fact storage across sessions.

Facts are extracted from conversation summaries and stored per character config.
On connect, they're injected into the system prompt so the AI remembers the user.
"""

import json
import os
from datetime import datetime, timezone
from typing import List, Optional
from loguru import logger

MEMORY_DIR = "memory"


def _memory_path(conf_uid: str) -> str:
    return os.path.join(MEMORY_DIR, conf_uid, "memories.json")


def _ensure_dir(conf_uid: str):
    dirpath = os.path.join(MEMORY_DIR, conf_uid)
    os.makedirs(dirpath, exist_ok=True)


class LongTermMemory:
    """Load, save, and manage persistent key facts for a character config."""

    def __init__(self, conf_uid: str):
        self.conf_uid = conf_uid
        self._facts: List[dict] = []
        self._last_summary_at: Optional[str] = None
        self._load()

    def _load(self):
        path = _memory_path(self.conf_uid)
        if not os.path.exists(path):
            self._facts = []
            self._last_summary_at = None
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._facts = data.get("facts", [])
            self._last_summary_at = data.get("last_summary_at")
            logger.info(f"Loaded {len(self._facts)} long-term memories for {self.conf_uid}")
        except Exception as e:
            logger.error(f"Failed to load memories for {self.conf_uid}: {e}")
            self._facts = []

    def save(self):
        _ensure_dir(self.conf_uid)
        path = _memory_path(self.conf_uid)
        data = {
            "facts": self._facts,
            "last_summary_at": self._last_summary_at,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self._facts)} memories for {self.conf_uid}")
        except Exception as e:
            logger.error(f"Failed to save memories for {self.conf_uid}: {e}")

    def get_facts(self) -> List[str]:
        """Return all fact strings."""
        return [f["text"] for f in self._facts]

    def get_facts_for_prompt(self) -> str:
        """Format facts as a prompt section."""
        facts = self.get_facts()
        if not facts:
            return ""
        lines = "\n".join(f"- {fact}" for fact in facts)
        return f"\n## What you remember about the user\n{lines}\n"

    def add_facts(self, new_facts: List[str]):
        """Add new facts, skipping near-duplicates."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        existing_texts = {f["text"].lower().strip() for f in self._facts}

        added = 0
        for fact in new_facts:
            normalized = fact.lower().strip()
            if normalized not in existing_texts and len(fact) > 5:
                self._facts.append({"text": fact, "added": today})
                existing_texts.add(normalized)
                added += 1

        if added:
            self._last_summary_at = datetime.now(timezone.utc).isoformat()
            self.save()
            logger.info(f"Added {added} new facts (skipped {len(new_facts) - added} duplicates)")

    def clear(self):
        """Clear all memories."""
        self._facts = []
        self._last_summary_at = None
        self.save()
        logger.info(f"Cleared all memories for {self.conf_uid}")

    @property
    def fact_count(self) -> int:
        return len(self._facts)
