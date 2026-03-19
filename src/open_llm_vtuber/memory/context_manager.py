"""
Context window manager with auto-summarization.

Prevents unbounded memory growth by maintaining a sliding window of recent
messages and summarizing older messages into persistent key facts.
"""

from typing import List, Dict, Any, Optional
from loguru import logger


class ContextManager:
    """Manages conversation context with sliding window and auto-summarization."""

    def __init__(
        self,
        max_recent_messages: int = 30,
        summarize_batch_size: int = 20,
        auto_summarize: bool = True,
    ):
        self.max_recent_messages = max_recent_messages
        self.summarize_batch_size = summarize_batch_size
        self.auto_summarize = auto_summarize
        self._pending_summary: Optional[List[Dict[str, Any]]] = None

    def check_needs_summary(self, memory: List[Dict[str, Any]]) -> bool:
        """Check if memory exceeds the window and needs summarization."""
        return len(memory) > self.max_recent_messages

    def extract_overflow(
        self, memory: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split memory into overflow (to summarize) and kept (recent) messages.

        Returns (overflow_messages, kept_messages).
        """
        if not self.check_needs_summary(memory):
            return [], memory

        overflow_count = min(self.summarize_batch_size, len(memory) - self.max_recent_messages)
        if overflow_count <= 0:
            return [], memory

        overflow = memory[:overflow_count]
        kept = memory[overflow_count:]
        return overflow, kept

    def format_for_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages into a readable transcript for the LLM summarizer."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    item.get("text", "") for item in content if item.get("type") == "text"
                ]
                content = " ".join(text_parts)
            label = "User" if role == "user" else "AI"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    async def summarize_messages(
        self,
        messages: List[Dict[str, Any]],
        llm,
        summary_prompt: str,
    ) -> List[str]:
        """Use the LLM to extract key facts from a batch of messages.

        Returns a list of fact strings.
        """
        transcript = self.format_for_summary(messages)
        prompt = f"{summary_prompt}\n\n--- Conversation segment ---\n{transcript}\n---\n\nKey facts:"

        llm_messages = [{"role": "user", "content": prompt}]

        facts = []
        full_response = ""
        try:
            async for chunk in llm.chat_completion(
                llm_messages,
                system="You extract key facts from conversations. Return only a bullet list.",
            ):
                if isinstance(chunk, str):
                    full_response += chunk

            for line in full_response.strip().split("\n"):
                line = line.strip().lstrip("-•* ").strip()
                if line and len(line) > 5:
                    facts.append(line)

            logger.info(f"Context manager extracted {len(facts)} facts from {len(messages)} messages")
        except Exception as e:
            logger.error(f"Failed to summarize messages: {e}")

        return facts

    def trim_memory(
        self, memory: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Hard-trim memory to max_recent_messages (no summarization, just drop)."""
        if len(memory) > self.max_recent_messages:
            trimmed = memory[-self.max_recent_messages:]
            logger.info(
                f"Hard-trimmed memory from {len(memory)} to {len(trimmed)} messages"
            )
            return trimmed
        return memory
