"""ACP agent server — exposes Hermes Agent via the Agent Client Protocol.

Supports multi-modal prompts (text, images, audio, resources).  Images are
passed through to AIAgent either as native ``image_url`` content blocks
(for models with vision) or described via auxiliary vision (for text-only
models) — mirroring the approach of the nanobot acp_adapter.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import tempfile
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Deque, Optional

import acp
from acp.schema import (
    AgentCapabilities,
    AuthenticateResponse,
    AvailableCommand,
    AvailableCommandsUpdate,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    ForkSessionResponse,
    ImageContentBlock,
    AudioContentBlock,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerHttp,
    McpServerSse,
    McpServerStdio,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    ResourceContentBlock,
    SessionCapabilities,
    SessionForkCapabilities,
    SessionListCapabilities,
    SessionResumeCapabilities,
    SessionInfo,
    TextContentBlock,
    UnstructuredCommandInput,
    Usage,
)

# AuthMethodAgent was renamed from AuthMethod in agent-client-protocol 0.9.0
try:
    from acp.schema import AuthMethodAgent
except ImportError:
    from acp.schema import AuthMethod as AuthMethodAgent  # type: ignore[attr-defined]

from acp_adapter.auth import detect_provider, has_provider
from acp_adapter.events import (
    make_message_cb,
    make_step_cb,
    make_thinking_cb,
    make_tool_progress_cb,
)
from acp_adapter.permissions import make_approval_callback
from acp_adapter.session import SessionManager, SessionState

logger = logging.getLogger(__name__)

try:
    from hermes_cli import __version__ as HERMES_VERSION
except Exception:
    HERMES_VERSION = "0.0.0"

# Thread pool for running AIAgent (synchronous) in parallel.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="acp-agent")

# ── Multi-modal prompt extraction ──────────────────────────────────────


def _extract_markdown_images(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Extract markdown image URLs from text.

    Returns:
        (text_without_images, [(alt_text, url), ...])
    """
    pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
    matches: list[tuple[str, str]] = re.findall(pattern, text)
    clean_text = re.sub(pattern, "", text).strip()
    return clean_text, matches


def _extract_text(
    prompt: list[
        TextContentBlock
        | ImageContentBlock
        | AudioContentBlock
        | ResourceContentBlock
        | EmbeddedResourceContentBlock
    ],
) -> str:
    """Extract plain text from ACP content blocks."""
    parts: list[str] = []
    for block in prompt:
        if isinstance(block, TextContentBlock):
            parts.append(block.text)
        elif hasattr(block, "text"):
            parts.append(str(block.text))
    return "\n".join(parts)


def _extract_prompt_parts(
    prompt: list[
        TextContentBlock
        | ImageContentBlock
        | AudioContentBlock
        | ResourceContentBlock
        | EmbeddedResourceContentBlock
    ],
) -> tuple[str, list[ImageContentBlock]]:
    """Extract text AND image blocks from an ACP prompt.

    Returns:
        (text, image_blocks) where image_blocks is a list of
        ImageContentBlock objects from the prompt.
    """
    text_parts: list[str] = []
    images: list[ImageContentBlock] = []

    for block in prompt:
        if isinstance(block, TextContentBlock):
            text_parts.append(block.text)
        elif isinstance(block, ImageContentBlock):
            images.append(block)
        elif hasattr(block, "text"):
            text_parts.append(str(getattr(block, "text", "")))

    return "\n".join(text_parts), images


async def _download_image(uri: str) -> tuple[str, str]:
    """Download an image from a URL to a temp file.

    Returns:
        (local_path, mime_type)
    """
    import httpx

    tmp_dir = Path(tempfile.gettempdir()) / "hermes_acp_images"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".jpg"
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(uri)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "png" in content_type:
            suffix = ".png"
        elif "gif" in content_type:
            suffix = ".gif"
        elif "webp" in content_type:
            suffix = ".webp"

        tmp_path = tmp_dir / f"acp_img_{os.urandom(8).hex()}{suffix}"
        tmp_path.write_bytes(resp.content)
        logger.info("Downloaded image from %s → %s (%d bytes)", uri, tmp_path, len(resp.content))
        return str(tmp_path), content_type or "image/jpeg"


def _materialize_base64_image(data: str, mime_type: str) -> str:
    """Write base64-encoded image data to a temp file.

    Returns:
        Local file path.
    """
    tmp_dir = Path(tempfile.gettempdir()) / "hermes_acp_images"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    suffix_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
    }
    suffix = suffix_map.get(mime_type, ".jpg")
    tmp_path = tmp_dir / f"acp_img_{os.urandom(8).hex()}{suffix}"

    if data.startswith("data:"):
        _, _, b64_data = data.partition(",")
    else:
        b64_data = data

    tmp_path.write_bytes(base64.b64decode(b64_data))
    logger.info("Materialized base64 image → %s (%d bytes)", tmp_path, tmp_path.stat().st_size)
    return str(tmp_path)


async def _describe_image_via_auxiliary(image_path: str) -> str:
    """Use the auxiliary vision router to describe an image.

    Falls back to a placeholder if the vision API is unavailable.
    Returns a text description to inject into the prompt.
    """
    try:
        from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
    except ImportError:
        logger.warning("Auxiliary vision not available, skipping image description")
        return f"[无法读取图片内容: {os.path.basename(image_path)}]"

    try:
        from tools.vision_tools import _image_to_base64_data_url, _detect_image_mime_type
        img_path = Path(image_path)
        mime = _detect_image_mime_type(img_path)
        if not mime:
            return f"[非图片文件: {os.path.basename(image_path)}]"

        data_url = _image_to_base64_data_url(img_path, mime_type=mime)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请用中文简要描述这张图片的内容。如果图片中有文字，请提取出来。"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        response = await async_call_llm(
            task="vision",
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            timeout=60,
        )
        description = extract_content_or_reasoning(response)
        if description:
            logger.info("Auxiliary vision description: %s", description[:100])
            return description
    except Exception:
        logger.warning("Auxiliary vision failed for %s", image_path, exc_info=True)

    return f"[图片识别失败: {os.path.basename(image_path)}]"


def _build_user_content_with_images(
    text: str, image_paths: list[tuple[str, str]]
) -> str | list[dict]:
    """Build user message content in OpenAI format with text and images.

    Args:
        text: User's text prompt
        image_paths: List of (local_path, mime_type) tuples

    Returns:
        Either a plain string (no images) or a list of content blocks
        compatible with OpenAI-format APIs.
    """
    if not image_paths:
        return text or "(用户发送了内容)"

    content: list[dict] = []
    if text:
        content.append({"type": "text", "text": text})

    for local_path, mime_type in image_paths:
        try:
            from tools.vision_tools import _image_to_base64_data_url
            data_url = _image_to_base64_data_url(Path(local_path), mime_type=mime_type)
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        except Exception:
            logger.warning("Failed to encode image %s for LLM", local_path, exc_info=True)

    return content if content else [{"type": "text", "text": "(空内容)"}]


class HermesACPAgent(acp.Agent):
    """ACP Agent implementation wrapping Hermes AIAgent."""

    _SLASH_COMMANDS = {
        "help": "Show available commands",
        "model": "Show or change current model",
        "tools": "List available tools",
        "context": "Show conversation context info",
        "reset": "Clear conversation history",
        "compact": "Compress conversation context",
        "version": "Show Hermes version",
    }

    _ADVERTISED_COMMANDS = (
        {
            "name": "help",
            "description": "List available commands",
        },
        {
            "name": "model",
            "description": "Show current model and provider, or switch models",
            "input_hint": "model name to switch to",
        },
        {
            "name": "tools",
            "description": "List available tools with descriptions",
        },
        {
            "name": "context",
            "description": "Show conversation message counts by role",
        },
        {
            "name": "reset",
            "description": "Clear conversation history",
        },
        {
            "name": "compact",
            "description": "Compress conversation context",
        },
        {
            "name": "version",
            "description": "Show Hermes version",
        },
    )

    def __init__(self, session_manager: SessionManager | None = None):
        super().__init__()
        self.session_manager = session_manager or SessionManager()
        self._conn: Optional[acp.Client] = None

    # ---- Connection lifecycle -----------------------------------------------

    def on_connect(self, conn: acp.Client) -> None:
        """Store the client connection for sending session updates."""
        self._conn = conn
        logger.info("ACP client connected")

    async def _register_session_mcp_servers(
        self,
        state: SessionState,
        mcp_servers: list[McpServerStdio | McpServerHttp | McpServerSse] | None,
    ) -> None:
        """Register ACP-provided MCP servers and refresh the agent tool surface."""
        if not mcp_servers:
            return

        try:
            from tools.mcp_tool import register_mcp_servers

            config_map: dict[str, dict] = {}
            for server in mcp_servers:
                name = server.name
                if isinstance(server, McpServerStdio):
                    config = {
                        "command": server.command,
                        "args": list(server.args),
                        "env": {item.name: item.value for item in server.env},
                    }
                else:
                    config = {
                        "url": server.url,
                        "headers": {item.name: item.value for item in server.headers},
                    }
                config_map[name] = config

            await asyncio.to_thread(register_mcp_servers, config_map)
        except Exception:
            logger.warning(
                "Session %s: failed to register ACP MCP servers",
                state.session_id,
                exc_info=True,
            )
            return

        try:
            from model_tools import get_tool_definitions

            enabled_toolsets = getattr(state.agent, "enabled_toolsets", None) or ["hermes-acp"]
            disabled_toolsets = getattr(state.agent, "disabled_toolsets", None)
            state.agent.tools = get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                quiet_mode=True,
            )
            state.agent.valid_tool_names = {
                tool["function"]["name"] for tool in state.agent.tools or []
            }
            invalidate = getattr(state.agent, "_invalidate_system_prompt", None)
            if callable(invalidate):
                invalidate()
            logger.info(
                "Session %s: refreshed tool surface after ACP MCP registration (%d tools)",
                state.session_id,
                len(state.agent.tools or []),
            )
        except Exception:
            logger.warning(
                "Session %s: failed to refresh tool surface after ACP MCP registration",
                state.session_id,
                exc_info=True,
            )

    # ---- ACP lifecycle ------------------------------------------------------

    async def initialize(
        self,
        protocol_version: int | None = None,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        resolved_protocol_version = (
            protocol_version if isinstance(protocol_version, int) else acp.PROTOCOL_VERSION
        )
        provider = detect_provider()
        auth_methods = None
        if provider:
            auth_methods = [
                AuthMethodAgent(
                    id=provider,
                    name=f"{provider} runtime credentials",
                    description=f"Authenticate Hermes using the currently configured {provider} runtime credentials.",
                )
            ]

        client_name = client_info.name if client_info else "unknown"
        logger.info(
            "Initialize from %s (protocol v%s)",
            client_name,
            resolved_protocol_version,
        )

        return InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(name="hermes-agent", version=HERMES_VERSION),
            agent_capabilities=AgentCapabilities(
                load_session=True,
                session_capabilities=SessionCapabilities(
                    fork=SessionForkCapabilities(),
                    list=SessionListCapabilities(),
                    resume=SessionResumeCapabilities(),
                ),
            ),
            auth_methods=auth_methods,
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        if has_provider():
            return AuthenticateResponse()
        return None

    # ---- Session management -------------------------------------------------

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        state = self.session_manager.create_session(cwd=cwd)
        await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("New session %s (cwd=%s)", state.session_id, cwd)
        self._schedule_available_commands_update(state.session_id)
        return NewSessionResponse(session_id=state.session_id)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        state = self.session_manager.update_cwd(session_id, cwd)
        if state is None:
            logger.warning("load_session: session %s not found", session_id)
            return None
        await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Loaded session %s", session_id)
        self._schedule_available_commands_update(session_id)
        return LoadSessionResponse()

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        state = self.session_manager.update_cwd(session_id, cwd)
        if state is None:
            logger.warning("resume_session: session %s not found, creating new", session_id)
            state = self.session_manager.create_session(cwd=cwd)
        await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Resumed session %s", state.session_id)
        self._schedule_available_commands_update(state.session_id)
        return ResumeSessionResponse()

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        state = self.session_manager.get_session(session_id)
        if state and state.cancel_event:
            state.cancel_event.set()
            try:
                if getattr(state, "agent", None) and hasattr(state.agent, "interrupt"):
                    state.agent.interrupt()
            except Exception:
                logger.debug("Failed to interrupt ACP session %s", session_id, exc_info=True)
            logger.info("Cancelled session %s", session_id)

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> ForkSessionResponse:
        state = self.session_manager.fork_session(session_id, cwd=cwd)
        new_id = state.session_id if state else ""
        if state is not None:
            await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Forked session %s -> %s", session_id, new_id)
        if new_id:
            self._schedule_available_commands_update(new_id)
        return ForkSessionResponse(session_id=new_id)

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        infos = self.session_manager.list_sessions()
        sessions = [
            SessionInfo(session_id=s["session_id"], cwd=s["cwd"])
            for s in infos
        ]
        return ListSessionsResponse(sessions=sessions)

    # ---- Prompt (core) ------------------------------------------------------

    async def prompt(
        self,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        session_id: str,
        **kwargs: Any,
    ) -> PromptResponse:
        """Run Hermes on the user's prompt and stream events back to the editor.

        Supports multi-modal prompts including images.  Images are handled via
        one of two paths:
          1. **Native vision**: for models that support ``image_url`` content
             blocks, the images are passed directly in the conversation history.
          2. **Auxiliary vision**: for text-only models, images are described
             using the auxiliary vision router and the descriptions are injected
             into the text prompt (matching nanobot acp_adapter behaviour).
        """
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.error("prompt: session %s not found", session_id)
            return PromptResponse(stop_reason="refusal")

        user_text, image_blocks = _extract_prompt_parts(prompt)
        user_text = user_text.strip()

        # ── Image handling ──────────────────────────────────────────────
        image_paths: list[tuple[str, str]] = []  # (local_path, mime_type)
        cleanup_paths: list[str] = []  # files to remove after this turn

        for img_block in image_blocks:
            try:
                if img_block.uri:
                    uri = img_block.uri
                    if uri.startswith("file://"):
                        local = uri[7:]
                        image_paths.append((local, img_block.mime_type))
                    else:
                        local, mime = await _download_image(uri)
                        image_paths.append((local, mime))
                        cleanup_paths.append(local)
                elif img_block.data:
                    local = _materialize_base64_image(img_block.data, img_block.mime_type)
                    image_paths.append((local, img_block.mime_type))
                    cleanup_paths.append(local)
            except Exception:
                logger.warning("Failed to process image block", exc_info=True)

        if image_paths:
            logger.info(
                "Prompt includes %d image(s): %s",
                len(image_paths),
                [os.path.basename(p) for p, _ in image_paths],
            )

        if not user_text and not image_paths:
            return PromptResponse(stop_reason="end_turn")

        if user_text.startswith("/"):
            response_text = self._handle_slash_command(user_text, state)
            if response_text is not None:
                if self._conn:
                    update = acp.update_agent_message_text(response_text)
                    await self._conn.session_update(session_id, update)
                return PromptResponse(stop_reason="end_turn")

        # ── Prepare user message content ────────────────────────────────
        model = getattr(state.agent, "model", "") or ""
        # Known text-only models (no vision capability)
        text_only_prefixes = ("glm-4-turbo", "glm-4-flash", "glm-4-air")
        has_vision = not any(model.lower().startswith(p) for p in text_only_prefixes)

        user_message_content: str | list[dict] = user_text or "(用户发送了内容)"

        if image_paths:
            if has_vision:
                user_message_content = _build_user_content_with_images(
                    user_text, image_paths
                )
                logger.info("Images passed as native content blocks to %s", model)
            else:
                for img_path, _mime in image_paths:
                    desc = await _describe_image_via_auxiliary(img_path)
                    if desc:
                        image_label = os.path.basename(img_path)
                        user_message_content = (
                            f"{user_message_content}\n\n"
                            f"[图片描述 ({image_label}): {desc}]"
                        )
                logger.info("Images described via auxiliary vision for %s", model)

        logger.info("Prompt on session %s: %s", session_id, user_text[:100] if user_text else "(image-only prompt)")

        conn = self._conn
        loop = asyncio.get_running_loop()

        if state.cancel_event:
            state.cancel_event.clear()

        tool_call_ids: dict[str, Deque[str]] = defaultdict(deque)
        previous_approval_cb = None

        if conn:
            tool_progress_cb = make_tool_progress_cb(conn, session_id, loop, tool_call_ids)
            thinking_cb = make_thinking_cb(conn, session_id, loop)
            step_cb = make_step_cb(conn, session_id, loop, tool_call_ids)
            message_cb = make_message_cb(conn, session_id, loop)
            approval_cb = make_approval_callback(conn.request_permission, loop, session_id)
        else:
            tool_progress_cb = None
            thinking_cb = None
            step_cb = None
            message_cb = None
            approval_cb = None

        agent = state.agent
        agent.tool_progress_callback = tool_progress_cb
        agent.thinking_callback = thinking_cb
        agent.step_callback = step_cb
        agent.message_callback = message_cb

        if approval_cb:
            try:
                from tools import terminal_tool as _terminal_tool
                previous_approval_cb = getattr(_terminal_tool, "_approval_callback", None)
                _terminal_tool.set_approval_callback(approval_cb)
            except Exception:
                logger.debug("Could not set ACP approval callback", exc_info=True)

        def _run_agent() -> dict:
            try:
                if isinstance(user_message_content, list):
                    user_msg = {"role": "user", "content": user_message_content}
                    state.history.append(user_msg)
                    result = agent.run_conversation(
                        user_message="",
                        conversation_history=state.history,
                        task_id=session_id,
                    )
                else:
                    result = agent.run_conversation(
                        user_message=user_message_content,
                        conversation_history=state.history,
                        task_id=session_id,
                    )
                return result
            except Exception as e:
                logger.exception("Agent error in session %s", session_id)
                return {"final_response": f"Error: {e}", "messages": state.history}
            finally:
                if approval_cb:
                    try:
                        from tools import terminal_tool as _terminal_tool
                        _terminal_tool.set_approval_callback(previous_approval_cb)
                    except Exception:
                        logger.debug("Could not restore approval callback", exc_info=True)

        try:
            result = await loop.run_in_executor(_executor, _run_agent)
        except Exception:
            logger.exception("Executor error for session %s", session_id)
            return PromptResponse(stop_reason="end_turn")
        finally:
            for fp in cleanup_paths:
                try:
                    os.unlink(fp)
                except OSError:
                    pass

        if result.get("messages"):
            state.history = result["messages"]
            self.session_manager.save_session(session_id)

        final_response = result.get("final_response", "")
        if final_response and conn:
            clean_text, images = _extract_markdown_images(final_response)
            if clean_text:
                update = acp.update_agent_message_text(clean_text)
                await conn.session_update(session_id, update)
            for alt, url in images:
                try:
                    if url.startswith(("http://", "https://")):
                        block = acp.image_block(data="", mime_type="image/jpeg", uri=url)
                    elif url.startswith("data:"):
                        header, _, data = url.partition(",")
                        mime = header.split(":", 1)[1].split(";", 1)[0] if ":" in header else "image/jpeg"
                        block = acp.image_block(data=data, mime_type=mime)
                    else:
                        continue
                    await conn.session_update(session_id, block)
                    logger.info("Sent outbound image: %s", url[:80])
                except Exception:
                    logger.warning("Failed to send image block for %s", url[:80], exc_info=True)

        usage = None
        if any(result.get(key) is not None for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
            usage = Usage(
                input_tokens=result.get("prompt_tokens", 0),
                output_tokens=result.get("completion_tokens", 0),
                total_tokens=result.get("total_tokens", 0),
                thought_tokens=result.get("reasoning_tokens"),
                cached_read_tokens=result.get("cache_read_tokens"),
            )

        stop_reason = "cancelled" if state.cancel_event and state.cancel_event.is_set() else "end_turn"
        return PromptResponse(stop_reason=stop_reason, usage=usage)

    # ---- Slash commands (headless) -------------------------------------------

    @classmethod
    def _available_commands(cls) -> list[AvailableCommand]:
        commands: list[AvailableCommand] = []
        for spec in cls._ADVERTISED_COMMANDS:
            input_hint = spec.get("input_hint")
            commands.append(
                AvailableCommand(
                    name=spec["name"],
                    description=spec["description"],
                    input=UnstructuredCommandInput(hint=input_hint)
                    if input_hint
                    else None,
                )
            )
        return commands

    async def _send_available_commands_update(self, session_id: str) -> None:
        """Advertise supported slash commands to the connected ACP client."""
        if not self._conn:
            return

        try:
            await self._conn.session_update(
                session_id=session_id,
                update=AvailableCommandsUpdate(
                    sessionUpdate="available_commands_update",
                    availableCommands=self._available_commands(),
                ),
            )
        except Exception:
            logger.warning(
                "Failed to advertise ACP slash commands for session %s",
                session_id,
                exc_info=True,
            )

    def _schedule_available_commands_update(self, session_id: str) -> None:
        """Send the command advertisement after the session response is queued."""
        if not self._conn:
            return
        loop = asyncio.get_running_loop()
        loop.call_soon(
            asyncio.create_task, self._send_available_commands_update(session_id)
        )

    def _handle_slash_command(self, text: str, state: SessionState) -> str | None:
        """Dispatch a slash command and return the response text.

        Returns ``None`` for unrecognized commands so they fall through
        to the LLM (the user may have typed ``/something`` as prose).
        """
        parts = text.split(maxsplit=1)
        cmd = parts[0].lstrip("/").lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        handler = {
            "help": self._cmd_help,
            "model": self._cmd_model,
            "tools": self._cmd_tools,
            "context": self._cmd_context,
            "reset": self._cmd_reset,
            "compact": self._cmd_compact,
            "version": self._cmd_version,
        }.get(cmd)

        if handler is None:
            return None  # not a known command — let the LLM handle it

        try:
            return handler(args, state)
        except Exception as e:
            logger.error("Slash command /%s error: %s", cmd, e, exc_info=True)
            return f"Error executing /{cmd}: {e}"

    def _cmd_help(self, args: str, state: SessionState) -> str:
        lines = ["Available commands:", ""]
        for cmd, desc in self._SLASH_COMMANDS.items():
            lines.append(f"  /{cmd:10s}  {desc}")
        lines.append("")
        lines.append("Unrecognized /commands are sent to the model as normal messages.")
        return "\n".join(lines)

    def _cmd_model(self, args: str, state: SessionState) -> str:
        if not args:
            model = state.model or getattr(state.agent, "model", "unknown")
            provider = getattr(state.agent, "provider", None) or "auto"
            return f"Current model: {model}\nProvider: {provider}"

        new_model = args.strip()
        target_provider = None
        current_provider = getattr(state.agent, "provider", None) or "openrouter"

        # Auto-detect provider for the requested model
        try:
            from hermes_cli.models import parse_model_input, detect_provider_for_model
            target_provider, new_model = parse_model_input(new_model, current_provider)
            if target_provider == current_provider:
                detected = detect_provider_for_model(new_model, current_provider)
                if detected:
                    target_provider, new_model = detected
        except Exception:
            logger.debug("Provider detection failed, using model as-is", exc_info=True)

        state.model = new_model
        state.agent = self.session_manager._make_agent(
            session_id=state.session_id,
            cwd=state.cwd,
            model=new_model,
            requested_provider=target_provider or current_provider,
        )
        self.session_manager.save_session(state.session_id)
        provider_label = getattr(state.agent, "provider", None) or target_provider or current_provider
        logger.info("Session %s: model switched to %s", state.session_id, new_model)
        return f"Model switched to: {new_model}\nProvider: {provider_label}"

    def _cmd_tools(self, args: str, state: SessionState) -> str:
        try:
            from model_tools import get_tool_definitions
            toolsets = getattr(state.agent, "enabled_toolsets", None) or ["hermes-acp"]
            tools = get_tool_definitions(enabled_toolsets=toolsets, quiet_mode=True)
            if not tools:
                return "No tools available."
            lines = [f"Available tools ({len(tools)}):"]
            for t in tools:
                name = t.get("function", {}).get("name", "?")
                desc = t.get("function", {}).get("description", "")
                # Truncate long descriptions
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                lines.append(f"  {name}: {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Could not list tools: {e}"

    def _cmd_context(self, args: str, state: SessionState) -> str:
        n_messages = len(state.history)
        if n_messages == 0:
            return "Conversation is empty (no messages yet)."
        # Count by role
        roles: dict[str, int] = {}
        for msg in state.history:
            role = msg.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1
        lines = [
            f"Conversation: {n_messages} messages",
            f"  user: {roles.get('user', 0)}, assistant: {roles.get('assistant', 0)}, "
            f"tool: {roles.get('tool', 0)}, system: {roles.get('system', 0)}",
        ]
        model = state.model or getattr(state.agent, "model", "")
        if model:
            lines.append(f"Model: {model}")
        return "\n".join(lines)

    def _cmd_reset(self, args: str, state: SessionState) -> str:
        state.history.clear()
        self.session_manager.save_session(state.session_id)
        return "Conversation history cleared."

    def _cmd_compact(self, args: str, state: SessionState) -> str:
        if not state.history:
            return "Nothing to compress — conversation is empty."
        try:
            agent = state.agent
            if not getattr(agent, "compression_enabled", True):
                return "Context compression is disabled for this agent."
            if not hasattr(agent, "_compress_context"):
                return "Context compression not available for this agent."

            from agent.model_metadata import estimate_messages_tokens_rough

            original_count = len(state.history)
            approx_tokens = estimate_messages_tokens_rough(state.history)
            original_session_db = getattr(agent, "_session_db", None)

            try:
                # ACP sessions must keep a stable session id, so avoid the
                # SQLite session-splitting side effect inside _compress_context.
                agent._session_db = None
                compressed, _ = agent._compress_context(
                    state.history,
                    getattr(agent, "_cached_system_prompt", "") or "",
                    approx_tokens=approx_tokens,
                    task_id=state.session_id,
                )
            finally:
                agent._session_db = original_session_db

            state.history = compressed
            self.session_manager.save_session(state.session_id)

            new_count = len(state.history)
            new_tokens = estimate_messages_tokens_rough(state.history)
            return (
                f"Context compressed: {original_count} -> {new_count} messages\n"
                f"~{approx_tokens:,} -> ~{new_tokens:,} tokens"
            )
        except Exception as e:
            return f"Compression failed: {e}"

    def _cmd_version(self, args: str, state: SessionState) -> str:
        return f"Hermes Agent v{HERMES_VERSION}"

    # ---- Model switching (ACP protocol method) -------------------------------

    async def set_session_model(
        self, model_id: str, session_id: str, **kwargs: Any
    ) -> SetSessionModelResponse | None:
        """Switch the model for a session (called by ACP protocol)."""
        state = self.session_manager.get_session(session_id)
        if state:
            state.model = model_id
            current_provider = getattr(state.agent, "provider", None)
            current_base_url = getattr(state.agent, "base_url", None)
            current_api_mode = getattr(state.agent, "api_mode", None)
            state.agent = self.session_manager._make_agent(
                session_id=session_id,
                cwd=state.cwd,
                model=model_id,
                requested_provider=current_provider,
                base_url=current_base_url,
                api_mode=current_api_mode,
            )
            self.session_manager.save_session(session_id)
            logger.info("Session %s: model switched to %s", session_id, model_id)
            return SetSessionModelResponse()
        logger.warning("Session %s: model switch requested for missing session", session_id)
        return None

    async def set_session_mode(
        self, mode_id: str, session_id: str, **kwargs: Any
    ) -> SetSessionModeResponse | None:
        """Persist the editor-requested mode so ACP clients do not fail on mode switches."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: mode switch requested for missing session", session_id)
            return None
        setattr(state, "mode", mode_id)
        self.session_manager.save_session(session_id)
        logger.info("Session %s: mode switched to %s", session_id, mode_id)
        return SetSessionModeResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        """Accept ACP config option updates even when Hermes has no typed ACP config surface yet."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: config update requested for missing session", session_id)
            return None

        options = getattr(state, "config_options", None)
        if not isinstance(options, dict):
            options = {}
        options[str(config_id)] = value
        setattr(state, "config_options", options)
        self.session_manager.save_session(session_id)
        logger.info("Session %s: config option %s updated", session_id, config_id)
        return SetSessionConfigOptionResponse(config_options=[])
