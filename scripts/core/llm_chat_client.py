"""
LLM 对话专用客户端 - 完全独立的 LLM 对话模型配置
支持思考模式（thinking）和高质量文档总结
"""
import base64
import logging
from typing import List, Optional, Dict, Any
import requests

from .config import Config

logger = logging.getLogger(__name__)


class LLMChatClient:
    """LLM 对话专用客户端 - 使用独立的 LLM 配置"""
    
    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        self.provider = (provider or Config.LLM_PROVIDER).lower()
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL
        self.thinking_enabled = Config.LLM_THINKING_ENABLED
        
        if not self.api_key:
            raise RuntimeError("LLM API key not configured. Set WORKDOCS_LLM_API_KEY in .env")
        
        # 设置 API endpoint
        if self.provider == "openai":
            self.chat_url = f"{self.base_url or 'https://api.openai.com/v1'}/chat/completions"
        elif self.provider == "kimi":
            self.chat_url = f"{self.base_url or 'https://api.moonshot.cn/v1'}/chat/completions"
        else:
            self.chat_url = f"{self.base_url}/chat/completions"
        
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
    
    def chat(self, messages: List[dict], temperature: float = 0.3, **kwargs) -> str:
        """基础对话功能"""
        data = {
            "model": self.model,
            "messages": messages
        }
        
        # 特殊处理 Kimi 模型限制
        if self.provider == "kimi" and self.model.startswith("kimi"):
            # Kimi 模型只支持 temperature=1.0
            data["temperature"] = 1.0
        else:
            data["temperature"] = temperature
        
        # 添加思考模式支持
        if self.thinking_enabled and "extra_body" not in kwargs:
            kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget": 1024}}
        
        # 合并额外参数
        data.update(kwargs)
        
        response_data = self._post(self.chat_url, data)
        return response_data["choices"][0]["message"]["content"]
    
    def chat_with_thinking(self, messages: List[dict], temperature: float = 0.3) -> str:
        """启用思考模式的对话"""
        # Kimi 模型不支持思考模式，回退到普通对话
        if self.provider == "kimi" and self.model.startswith("kimi"):
            logger.warning("Kimi 模型不支持思考模式，回退到普通对话")
            return self.chat(messages)
        return self.chat(messages, temperature, extra_body={"thinking": {"type": "enabled", "budget": 1024}})
    
    def summarize(self, text: str, max_tokens: int = 1000, prompt_template: Optional[str] = None) -> dict:
        """智能文本总结"""
        system_prompt = """你是一个技术文档分析专家。请对以下内容进行结构化总结，格式如下：

Summary: [用中文提供简洁的技术总结，200-300字]

Keywords: [关键词1, 关键词2, 关键词3]

请确保总结准确、专业，突出技术要点。"""
        
        user_prompt = prompt_template or "请总结以下技术文档内容：\n\n{{text}}"
        content = user_prompt.replace("{{text}}", text[:12000])  # 截断避免超限
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        
        # Kimi 模型使用固定 temperature=1.0
        if self.provider == "kimi" and self.model.startswith("kimi"):
            raw = self.chat(messages)  # 自动使用 temperature=1.0
        else:
            raw = self.chat(messages, temperature=0.3)
        
        # 解析结构化输出
        summary = ""
        keywords = []
        
        for line in raw.splitlines():
            if line.lower().startswith("summary:"):
                summary = line.split(":", 1)[1].strip()
            elif line.lower().startswith("keywords:"):
                kw_part = line.split(":", 1)[1].strip()
                keywords = [k.strip() for k in kw_part.split(",") if k.strip()]
        
        # 回退处理
        if not summary:
            summary = raw.strip()
        
        return {
            "summary": summary,
            "keywords": keywords,
            "raw": raw
        }
    
    def vision_describe(self, image_path: str, prompt: str = "详细描述这个技术图表的内容和含义。") -> str:
        """图像分析和描述"""
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
        
        return self.chat(messages, temperature=0.3)
    
    def hierarchical_summarize(self, texts: List[str], max_tokens_per_chunk: int = 800) -> str:
        """层次化总结多个文本块"""
        if not texts:
            return ""
        
        if len(texts) == 1:
            return self.summarize(texts[0], max_tokens=max_tokens_per_chunk)["summary"]
        
        # 第一层：块级总结
        chunk_summaries = []
        for i, text in enumerate(texts):
            summary = self.summarize(text, max_tokens=max_tokens_per_chunk)["summary"]
            chunk_summaries.append(f"部分{i+1}: {summary}")
        
        # 第二层：总结汇总
        combined = "\n\n".join(chunk_summaries)
        return self.summarize(combined, max_tokens=max_tokens_per_chunk * 2)["summary"]
    
    def close(self) -> None:
        """关闭会话"""
        self._session.close()
    
    def close(self) -> None:
        """关闭会话"""
        self._session.close()