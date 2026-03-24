from typing import Union, List, Dict, Any, Optional
import asyncio
import json
import re
from loguru import logger
import numpy as np

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    EMOJI_LIST,
)
from .toggle_tag_parser import (
    parse_and_send_toggle_tags,
    extract_sentence_toggle_tags,
    normalize_tags,
    strip_unknown_tags,
    all_bracket_tags_for_display_strip,
    BODY_MOTION_COMMANDS,
    CLOTHING_TOGGLE_COMMANDS,
    MAX_BODY_MOTIONS_PER_SENTENCE,
)
from ..agent.output_types import Actions
from .types import WebSocketSend
from .tts_manager import TTSTaskManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext

from ..agent.output_types import SentenceOutput, AudioOutput


async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Process a single-user conversation turn

    Args:
        context: Service context containing all configurations and engines
        websocket_send: WebSocket send function
        client_uid: Client unique identifier
        user_input: Text or audio input from user
        images: Optional list of image data
        session_emoji: Emoji identifier for the conversation
        metadata: Optional metadata for special processing flags

    Returns:
        str: Complete response text
    """
    # Create TTSTaskManager for this conversation
    tts_manager = TTSTaskManager()
    full_response = ""  # Initialize full_response here

    try:
        # Send initial signals
        await send_conversation_start_signals(websocket_send)
        logger.info(f"New Conversation Chain {session_emoji} started!")

        # Process user input
        input_text = await process_user_input(
            user_input, context.asr_engine, websocket_send
        )

        # Create batch input
        batch_input = create_batch_input(
            input_text=input_text,
            images=images,
            from_name=context.character_config.human_name,
            metadata=metadata,
        )

        # Store user message (check if we should skip storing to history)
        skip_history = metadata and metadata.get("skip_history", False)
        if context.history_uid and not skip_history:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="human",
                content=input_text,
                name=context.character_config.human_name,
            )

        if skip_history:
            logger.debug("Skipping storing user input to history (proactive speak)")

        logger.info(f"User input: {input_text}")
        if images:
            logger.info(f"With {len(images)} images")

        try:
            # agent.chat yields Union[SentenceOutput, Dict[str, Any]]
            agent_output_stream = context.agent_engine.chat(batch_input)
            clothing_toggles_sent: set[str] = set()

            async for output_item in agent_output_stream:
                if (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "tool_call_status"
                ):
                    output_item["name"] = context.character_config.character_name
                    logger.debug(f"Sending tool status update: {output_item}")

                    await websocket_send(json.dumps(output_item))

                elif isinstance(output_item, (SentenceOutput, AudioOutput)):
                    if isinstance(output_item, SentenceOutput):
                        output_item.display_text.text = normalize_tags(
                            output_item.display_text.text
                        )
                        _, toggle_cmds = extract_sentence_toggle_tags(
                            output_item.display_text.text,
                            context.affection,
                        )
                        output_item.display_text.text = strip_unknown_tags(
                            output_item.display_text.text,
                            known_tags=all_bracket_tags_for_display_strip(
                                set(context.live2d_model.emo_map.keys())
                            ),
                        )
                        if toggle_cmds:
                            kept = []
                            sentence_motion_count = 0
                            for cmd in toggle_cmds:
                                if cmd in BODY_MOTION_COMMANDS:
                                    if sentence_motion_count < MAX_BODY_MOTIONS_PER_SENTENCE:
                                        kept.append(cmd)
                                        sentence_motion_count += 1
                                    else:
                                        logger.info(
                                            f"Body-motion cap reached "
                                            f"({MAX_BODY_MOTIONS_PER_SENTENCE}), "
                                            f"dropping {cmd}"
                                        )
                                elif cmd in CLOTHING_TOGGLE_COMMANDS:
                                    if cmd not in clothing_toggles_sent:
                                        kept.append(cmd)
                                        clothing_toggles_sent.add(cmd)
                                    else:
                                        logger.info(
                                            f"Duplicate clothing toggle, "
                                            f"dropping repeat {cmd}"
                                        )
                                else:
                                    kept.append(cmd)
                            if kept:
                                if output_item.actions is None:
                                    output_item.actions = Actions()
                                output_item.actions.toggles = kept

                    response_part = await process_agent_output(
                        output=output_item,
                        character_config=context.character_config,
                        live2d_model=context.live2d_model,
                        tts_engine=context.tts_engine,
                        websocket_send=websocket_send,  # Pass websocket_send for audio/tts messages
                        tts_manager=tts_manager,
                        translate_engine=context.translate_engine,
                    )
                    # Ensure response_part is treated as a string before concatenation
                    response_part_str = (
                        str(response_part) if response_part is not None else ""
                    )
                    full_response += response_part_str  # Accumulate text response
                else:
                    logger.warning(
                        f"Received unexpected item type from agent chat stream: {type(output_item)}"
                    )
                    logger.debug(f"Unexpected item content: {output_item}")

        except Exception as e:
            logger.exception(
                f"Error processing agent response stream: {e}"
            )  # Log with stack trace
            await websocket_send(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error processing agent response: {str(e)}",
                    }
                )
            )
            # full_response will contain partial response before error
        # --- End processing agent response ---

        # Wait for any pending TTS tasks
        if tts_manager.task_list:
            await asyncio.gather(*tts_manager.task_list)
            try:
                await websocket_send(
                    json.dumps({"type": "backend-synth-complete"})
                )
            except Exception:
                logger.warning(
                    "WebSocket error sending backend-synth-complete, "
                    "continuing conversation cleanup"
                )

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            client_uid=client_uid,
        )

        if full_response and context.affection:
            full_response = await parse_and_send_toggle_tags(
                full_response, websocket_send, context.affection
            )

        if context.history_uid and full_response:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=full_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            logger.info(f"AI response: {full_response}")

        return full_response  # Return accumulated full_response

    except asyncio.CancelledError:
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        logger.error(f"Error in conversation chain: {e}")
        try:
            await websocket_send(
                json.dumps(
                    {"type": "error", "message": f"Conversation error: {str(e)}"}
                )
            )
        except Exception:
            logger.warning("WebSocket broken, could not send error to client")
        raise
    finally:
        cleanup_conversation(tts_manager, session_emoji)
