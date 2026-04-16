import base64
from typing import List, Optional

import requests

from .config import Config


class _BaseClient:
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


class ChatClient(_BaseClient):
    def chat(self, messages: List[dict], temperature: float = 0.3) -> str:
        data = self._post(self.chat_url, {"model": self.model, "messages": messages, "temperature": temperature})
        return data["choices"][0]["message"]["content"]

    def summarize(self, text: str, prompt_template: Optional[str] = None) -> dict:
        tpl = prompt_template or self._load_prompt("summarize")
        content = tpl.replace("{{text}}", text[:8000])
        raw = self.chat([{"role": "user", "content": content}])
        summary = ""
        keywords: List[str] = []
        for line in raw.splitlines():
            if line.lower().startswith("summary:"):
                summary = line.split(":", 1)[1].strip()
            elif line.lower().startswith("keywords:"):
                kw_part = line.split(":", 1)[1].strip()
                keywords = [k.strip() for k in kw_part.split(",") if k.strip()]
        return {"summary": summary or raw.strip(), "keywords": keywords}

    def vision_describe(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = image_path.split(".")[-1].lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png" if ext == "png" else "image/webp"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ]
        return self.chat(messages)

    @staticmethod
    def _load_prompt(name: str) -> str:
        path = Config.PROMPT_DIR / f"{name}.txt"
        return path.read_text(encoding="utf-8") if path.exists() else "Summarize the following text:\n{{text}}"


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
