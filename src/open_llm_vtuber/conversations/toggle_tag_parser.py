"""
Parse affection-gated toggle tags from LLM responses.

When the AI outputs tags like [hideclothes], [hideunderwear], etc., this module
strips them from the response text and sends the corresponding toggle-parameter
messages to the frontend -- but only if the current affection tier allows it.
"""

import re
import json
from loguru import logger

TAG_TO_COMMAND = {
    "hideclothes": "/hideclothes",
    "hidecover": "/hidecover",
    "hideunderwear": "/hideunderwear",
    "hidex": "/hidex",
    "reset": "/reset",
    "nod": "/yes",
    "shake": "/no",
    "tilt": "/tilt",
    "confused": "/confused",
    "dance": "/dance",
    "excited": "/excited",
    "wink": "/wink",
    "think": "/think",
}

ALWAYS_ALLOWED_TAGS = {
    "nod", "shake", "tilt", "confused", "reset",
    "dance", "excited", "wink", "think",
}

TAG_PATTERN = re.compile(
    r"\[(" + "|".join(TAG_TO_COMMAND.keys()) + r")\]",
    re.IGNORECASE,
)


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
            await websocket_send(json.dumps({
                "type": "toggle-parameter",
                "command": command,
            }))
        else:
            logger.debug(
                f"Toggle [{tag_name}] not unlocked at current tier "
                f"(unlocked: {unlocked})"
            )

    cleaned = TAG_PATTERN.sub("", response_text).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned
