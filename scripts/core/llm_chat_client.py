"""LLM 对话客户端.

基于统一 APIClient 构建，按 Kimi 官方错误码分类处理并重试。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .api_client import APIClient, ContentTooLargeError, KimiProvider
from .config import Config

logger = logging.getLogger(__name__)


class BaseLLMClient:
    """LLM 对话客户端."""

    # LLM 同步对话请求的超时；重试统一由 APIClient（HTTP_RETRY_* 配置）处理。
    DEFAULT_TIMEOUT = Config.LLM_TIMEOUT

    @staticmethod
    def _load_user_agent() -> str:
        """从 plugin.json 读取 runtime 信息构建 User-Agent.

        Kimi Coding API 白名单要求 User-Agent 前缀为 KimiCLI/ 才能通过验证.
        """
        plugin_path = Path(__file__).resolve().parent.parent.parent / "kimi.plugin.json"
        if plugin_path.exists():
            try:
                data = json.loads(plugin_path.read_text(encoding="utf-8"))
                version = data.get("runtime", {}).get("host_version", "")
                if version:
                    return f"KimiCLI/{version}"
            except Exception:
                pass
        return "KimiCLI/1.44.0"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        """初始化 BaseLLMClient."""
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL
        self.thinking_enabled = Config.LLM_THINKING_ENABLED
        self.user_agent = user_agent or self._load_user_agent()

        if not self.api_key:
            raise RuntimeError("LLM API key not configured. Set WORKDOCS_LLM_API_KEY in .env")

        provider = KimiProvider(api_key=self.api_key, base_url=self.base_url)
        self._client = APIClient(provider, timeout=self.DEFAULT_TIMEOUT, user_agent=self.user_agent)

    def _post(self, url: str, payload: dict, timeout: int | None = None) -> dict:
        """发送 POST 请求.

        保留此接口以便现有调用点与部分单测 monkeypatch 兼容。
        实际请求由 APIClient 处理，包含统一的错误分类与重试。
        url 参数为相对 path（APIClient 会拼接 base_url），例如 `/chat/completions`。
        """
        messages = payload.get("messages", [])
        text_len = sum(len(str(m.get("content", ""))) for m in messages)
        img_count = sum(
            1
            for m in messages
            if isinstance(m.get("content"), list)
            for item in m["content"]
            if isinstance(item, dict) and item.get("type") == "image_url"
        )
        logger.info(f"LLM 请求开始 | text_len={text_len} | images={img_count}")
        start_time = time.time()

        original_timeout = self._client.timeout
        if timeout is not None:
            self._client.timeout = timeout
        try:
            response = self._client.post(url, json=payload)
        finally:
            self._client.timeout = original_timeout

        elapsed = time.time() - start_time
        logger.info(f"LLM 请求成功 | elapsed={elapsed:.1f}s | status={response.status_code}")
        return response.json()

    def chat(self, messages: list[dict], **kwargs) -> str:
        """基础对话功能."""
        data = {"model": self.model, "messages": messages}

        # 添加思考模式支持（OpenAI 兼容格式，始终传递确保模型行为可控）
        thinking_type = "enabled" if self.thinking_enabled else "disabled"
        extra_body = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("thinking", {"type": thinking_type})

        # 合并额外参数
        data.update(kwargs)

        try:
            response_data = self._post(Config.LLM_CHAT_ENDPOINT, data)
        except ContentTooLargeError as exc:
            total_len = sum(len(str(m.get("content", ""))) for m in messages)
            logger.warning(f"LLM 请求内容超长 | text_len={total_len} | message={exc.message!r}")
            raise

        # 防御性校验 API 响应结构
        if not isinstance(response_data, dict):
            raise RuntimeError("Invalid response: not a dict")
        choices = response_data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise RuntimeError("Invalid response: missing or empty choices")
        message = choices[0].get("message")
        if not message or not isinstance(message, dict):
            raise RuntimeError("Invalid response: missing message")
        content = message.get("content")
        if content is None:
            raise RuntimeError("Invalid response: missing content")
        return content

    def close(self) -> None:
        """关闭底层 HTTP 客户端."""
        self._client.close()
