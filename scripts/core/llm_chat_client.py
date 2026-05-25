"""LLM 对话客户端 - 基类.

支持同步对话调用.
"""

import json
import logging
import threading
from pathlib import Path

import requests

from .config import Config

logger = logging.getLogger(__name__)


class BaseLLMClient:
    """LLM 对话客户端基类."""

    # 类常量 - 从 Config 读取，保留属性供测试覆盖
    MAX_RETRY_ATTEMPTS = Config.LLM_MAX_RETRIES
    RETRY_BACKOFF_BASE = Config.LLM_RETRY_BACKOFF
    DEFAULT_TIMEOUT = Config.LLM_TIMEOUT

    @staticmethod
    def _load_user_agent() -> str:
        """从 plugin.json 读取 runtime 信息构建 User-Agent.

        Kimi Coding API 白名单要求 User-Agent 前缀为 KimiCLI/ 才能通过验证.
        """
        plugin_path = Path(__file__).resolve().parent.parent.parent / "plugin.json"
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

        # 完全由 base_url 决定，不做服务商推断
        self.chat_url = f"{self.base_url}/chat/completions"
        # embed_url 保留以兼容 LLMClient 多重继承
        self.embed_url = f"{self.base_url}/embeddings"

        # 使用线程本地存储，确保多线程并发时每个线程有独立的 Session
        self._local = threading.local()

    def _get_session(self) -> requests.Session:
        """获取当前线程的 requests.Session（懒创建）."""
        if not hasattr(self._local, "session") or self._local.session is None:
            self._local.session = requests.Session()
        return self._local.session

    def _post(self, url: str, payload: dict, timeout: int | None = None) -> dict:
        """发送 POST 请求，带重试机制.

        仅对可重试错误（5xx、429）执行指数退避重试，
        4xx 客户端错误（除 429 外）直接抛出，
        超时异常直接抛出（重试使用相同 timeout 无法解决请求过大的超时问题）。
        """
        import time

        timeout = timeout or self.DEFAULT_TIMEOUT
        session = self._get_session()
        last_exc = None

        # 计算请求大小用于日志和诊断
        messages = payload.get("messages", [])
        text_len = sum(len(str(m.get("content", ""))) for m in messages)
        img_count = sum(
            1
            for m in messages
            if isinstance(m.get("content"), list)
            for item in m["content"]
            if isinstance(item, dict) and item.get("type") == "image_url"
        )
        logger.info(
            f"LLM 请求开始 | text_len={text_len} | images={img_count} | timeout={timeout}s"
        )
        start_time = time.time()

        for attempt in range(self.MAX_RETRY_ATTEMPTS):
            try:
                resp = session.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": self.user_agent,
                    },
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                elapsed = time.time() - start_time
                logger.info(f"LLM 请求成功 | elapsed={elapsed:.1f}s | status=ok")
                return resp.json()
            except requests.Timeout as e:
                elapsed = time.time() - start_time
                suggested = max(timeout * 2, 300)
                logger.warning(
                    f"LLM 请求超时 | elapsed={elapsed:.1f}s | timeout={timeout}s | "
                    f"text_len={text_len} | images={img_count} | "
                    f"建议增大 WORKDOCS_LLM_TIMEOUT 至 {suggested}s 以上"
                )
                raise RuntimeError(
                    f"LLM 请求超时 ({timeout}s)。请求大小: text_len={text_len}, "
                    f"images={img_count}。建议: export WORKDOCS_LLM_TIMEOUT={suggested} "
                    f"或在 config.json 中设置 'llm.timeout'={suggested}"
                ) from e
            except requests.HTTPError as e:
                last_exc = e
                status_code = e.response.status_code if e.response else 0
                # 4xx 客户端错误（除 429 Too Many Requests 外）不可重试
                if 400 <= status_code < 500 and status_code != 429:
                    raise
                time.sleep(self.RETRY_BACKOFF_BASE**attempt)
            except Exception as e:
                last_exc = e
                time.sleep(self.RETRY_BACKOFF_BASE**attempt)
        assert last_exc is not None
        raise last_exc

    def chat(self, messages: list[dict], temperature: float = 0.3, **kwargs) -> str:
        """基础对话功能."""
        data = {"model": self.model, "messages": messages, "temperature": temperature}

        # 添加思考模式支持（OpenAI 兼容格式，始终传递确保模型行为可控）
        thinking_type = "enabled" if self.thinking_enabled else "disabled"
        extra_body = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("thinking", {"type": thinking_type})

        # 合并额外参数
        data.update(kwargs)

        response_data = self._post(self.chat_url, data)
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
        """关闭当前线程的会话."""
        if hasattr(self._local, "session") and self._local.session is not None:
            self._local.session.close()
            self._local.session = None

    @staticmethod
    def _load_prompt(name: str) -> str:
        """从 prompts/ 目录加载提示词文件，若不存在则返回内置默认值."""
        path = Config.PROMPT_DIR / f"{name}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""
