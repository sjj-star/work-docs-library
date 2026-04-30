"""Embedding 专用客户端 - 完全独立的 Embedding 模型配置.

与 LLM 对话模型使用不同的 API 密钥和端点.
"""

import logging

import requests

from .config import Config

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Embedding 专用客户端 - 使用独立的 Embedding 配置."""

    # 类常量 - 从 Config 读取，保留属性供测试覆盖
    MAX_RETRY_ATTEMPTS = Config.EMBED_MAX_RETRIES
    RETRY_BACKOFF_BASE = Config.EMBED_RETRY_BACKOFF
    DEFAULT_TIMEOUT = Config.EMBED_TIMEOUT
    MAX_BATCH_SIZE_LIMIT = Config.EMBED_MAX_BATCH_SIZE

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """初始化 EmbeddingClient."""
        self.api_key = api_key or Config.EMBEDDING_API_KEY
        self.base_url = base_url or Config.EMBEDDING_BASE_URL
        self.model = model or Config.EMBEDDING_MODEL

        # 运行时探测的实际维度（首次调用后设置）
        self._actual_dimension: int | None = None
        self._dim_validated = False

        if not self.api_key:
            raise RuntimeError(
                "Embedding API key not configured. Set WORKDOCS_EMBEDDING_API_KEY in .env"
            )

        # 设置 API endpoint（完全由 base_url 决定，不做服务商推断）
        self.embed_url = f"{self.base_url}/embeddings"

        self._session = requests.Session()

    def _post(self, url: str, payload: dict, timeout: int | None = None) -> dict:
        """发送 POST 请求，带重试机制."""
        timeout = timeout or self.DEFAULT_TIMEOUT
        last_exc = None
        for attempt in range(self.MAX_RETRY_ATTEMPTS):
            try:
                resp = self._session.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_exc = e
                import time

                time.sleep(self.RETRY_BACKOFF_BASE**attempt)
        assert last_exc is not None
        raise last_exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        """生成文本嵌入向量."""
        if not texts:
            return []

        # 分批处理避免 API 限制
        batch_size = min(Config.BATCH_SIZE, self.MAX_BATCH_SIZE_LIMIT)
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # 构建请求数据
            request_data = {"model": self.model, "input": batch_texts}

            # 无条件添加 dimensions 参数
            # 支持的 API 会生效，不支持的 API 会忽略
            config_dim = int(Config.EMBEDDING_DIMENSION)
            if config_dim > 0:
                request_data["dimensions"] = config_dim

            data = self._post(self.embed_url, request_data)

            items = data["data"]
            items.sort(key=lambda x: x["index"])
            batch_embeddings = [item["embedding"] for item in items]
            all_embeddings.extend(batch_embeddings)

        # 首次调用：验证维度
        if not self._dim_validated and all_embeddings:
            actual_dim = len(all_embeddings[0])
            expected_dim = Config.EMBEDDING_DIMENSION

            if actual_dim != expected_dim:
                logger.warning(
                    f"Embedding dimension mismatch: API returned {actual_dim} dimensions, "
                    f"but configuration specifies {expected_dim}. "
                    f"Please update WORKDOCS_EMBEDDING_DIMENSION to {actual_dim} "
                    f"or verify your model supports the configured dimension."
                )

            self._actual_dimension = actual_dim
            self._dim_validated = True
            logger.info(f"Embedding dimension detected and locked: {actual_dim}")

        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """生成单个文本的嵌入向量."""
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else []

    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度（首次调用后可用）."""
        if self._actual_dimension is None:
            raise RuntimeError(
                "Embedding dimension not yet detected. Call embed() at least once first."
            )
        return self._actual_dimension

    def close(self) -> None:
        """关闭会话."""
        self._session.close()
