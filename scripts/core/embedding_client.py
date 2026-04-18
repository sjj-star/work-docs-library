"""
Embedding 专用客户端 - 完全独立的 Embedding 模型配置
与 LLM 对话模型使用不同的 API 密钥和端点
"""
from typing import List, Optional
import requests

from .config import Config


class EmbeddingClient:
    """Embedding 专用客户端 - 使用独立的 Embedding 配置"""
    
    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        self.provider = (provider or Config.EMBEDDING_PROVIDER).lower()
        self.api_key = api_key or Config.EMBEDDING_API_KEY
        self.base_url = base_url or Config.EMBEDDING_BASE_URL
        self.model = model or Config.EMBEDDING_MODEL
        self.embedding_dim = Config.EMBEDDING_DIM
        
        if not self.api_key:
            raise RuntimeError("Embedding API key not configured. Set WORKDOCS_EMBEDDING_API_KEY in .env")
        
        # 设置 API endpoint
        if self.provider == "openai":
            self.embed_url = f"{self.base_url or 'https://api.openai.com/v1'}/embeddings"
        elif self.provider == "kimi":
            self.embed_url = f"{self.base_url or 'https://api.moonshot.cn/v1'}/embeddings"
        else:
            self.embed_url = f"{self.base_url}/embeddings"
        
        self._session = requests.Session()
    
    def _post(self, url: str, payload: dict, timeout: int = 120) -> dict:
        """发送 POST 请求，带重试机制"""
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
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入向量"""
        if not texts:
            return []
        
        # 分批处理避免 API 限制
        batch_size = min(Config.BATCH_SIZE, 100)  # 大多数 API 的限制
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            data = self._post(self.embed_url, {
                "model": self.model,
                "input": batch_texts
            })
            
            items = data["data"]
            items.sort(key=lambda x: x["index"])
            batch_embeddings = [item["embedding"] for item in items]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_single(self, text: str) -> List[float]:
        """生成单个文本的嵌入向量"""
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else []
    
    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        return self.embedding_dim
    
    def close(self) -> None:
        """关闭会话"""
        self._session.close()