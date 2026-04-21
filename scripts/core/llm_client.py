import base64
from typing import List, Optional

import requests

from .config import Config
from .llm_chat_client import LLMChatClient


class _BaseClient:
    """向后兼容的基类（供 EmbeddingClient 使用）"""
    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.provider = (provider or Config.LLM_PROVIDER).lower()
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = Config.LLM_MODEL
        self.embedding_model = Config.EMBEDDING_MODEL

        if not self.api_key:
            raise RuntimeError("LLM API key not configured. Set WORKDOCS_LLM_API_KEY in .env")

        if self.provider == "openai":
            self.chat_url = f"{self.base_url or 'https://api.openai.com/v1'}/chat/completions"
            self.embed_url = f"{self.base_url or 'https://api.openai.com/v1'}/embeddings"
        elif self.provider == "kimi":
            self.chat_url = f"{self.base_url or 'https://api.moonshot.cn/v1'}/chat/completions"
            self.embed_url = f"{self.base_url or 'https://api.moonshot.cn/v1'}/embeddings"
        else:
            self.chat_url = f"{self.base_url}/chat/completions"
            self.embed_url = f"{self.base_url}/embeddings"

        self._session = requests.Session()

    def _post(self, url: str, payload: dict, timeout: int = 120) -> dict:
        last_exc = None
        for attempt in range(3):
            try:
                resp = self._session.post(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_exc = e
                import time
                time.sleep(2 ** attempt)
        raise last_exc

    def close(self) -> None:
        self._session.close()


class ChatClient(LLMChatClient):
    """Backward-compatible chat client. Inherits from LLMChatClient."""
    pass


class EmbeddingClient(_BaseClient):
    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        data = self._post(self.embed_url, {"model": self.embedding_model, "input": texts})
        items = data["data"]
        items.sort(key=lambda x: x["index"])
        return [item["embedding"] for item in items]


class LLMClient(ChatClient, EmbeddingClient):
    """Backward-compatible combined client."""
    pass
