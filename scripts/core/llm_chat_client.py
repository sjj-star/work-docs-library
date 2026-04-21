"""
LLM 对话客户端 - 基类 + 增强子类
支持思考模式（thinking）和高质量文档总结
"""
import base64
import logging
from pathlib import Path
from typing import List, Optional
import requests

from .config import Config

logger = logging.getLogger(__name__)


class BaseLLMClient:
    """LLM 对话客户端基类"""

    # 类常量 - Magic Number 提取
    MAX_RETRY_ATTEMPTS = 3
    RETRY_BACKOFF_BASE = 2
    DEFAULT_TIMEOUT = 120
    MAX_INPUT_CHARS = 12000
    THINKING_BUDGET = 1024
    DEFAULT_SUMMARY_MAX_TOKENS = 1000
    SUMMARY_TARGET_LENGTH = 250

    # MIME 类型映射
    IMAGE_MIME_TYPES = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp"
    }
    DEFAULT_IMAGE_MIME = "image/webp"

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        self.provider = (provider or Config.LLM_PROVIDER).lower()
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL
        self.thinking_enabled = Config.LLM_THINKING_ENABLED

        if not self.api_key:
            raise RuntimeError("LLM API key not configured. Set WORKDOCS_LLM_API_KEY in .env")

        if self.provider == "openai":
            base = self.base_url or "https://api.openai.com/v1"
        elif self.provider == "kimi":
            base = self.base_url or "https://api.moonshot.cn/v1"
        else:
            base = self.base_url

        self.chat_url = f"{base}/chat/completions"
        # embed_url 保留以兼容 LLMClient 多重继承
        self.embed_url = f"{base}/embeddings"

        self._session = requests.Session()

    def _post(self, url: str, payload: dict, timeout: int = None) -> dict:
        """发送 POST 请求，带重试机制"""
        timeout = timeout or self.DEFAULT_TIMEOUT
        last_exc = None
        for attempt in range(self.MAX_RETRY_ATTEMPTS):
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
                time.sleep(self.RETRY_BACKOFF_BASE ** attempt)
        raise last_exc

    def chat(self, messages: List[dict], temperature: float = 0.3, **kwargs) -> str:
        """基础对话功能"""
        data = {
            "model": self.model,
            "messages": messages
        }

        # 特殊处理 Kimi 模型限制
        if self.provider == "kimi" and self.model.startswith("kimi"):
            data["temperature"] = 1.0
        else:
            data["temperature"] = temperature

        # 添加思考模式支持
        if self.thinking_enabled and "extra_body" not in kwargs:
            kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget": self.THINKING_BUDGET}}

        # 合并额外参数
        data.update(kwargs)

        response_data = self._post(self.chat_url, data)
        return response_data["choices"][0]["message"]["content"]

    def summarize(self, text: str, max_tokens: int = None, prompt_template: Optional[str] = None) -> dict:
        """智能文本总结"""
        max_tokens = max_tokens or self.DEFAULT_SUMMARY_MAX_TOKENS

        system_prompt = self._load_prompt("summarize")
        user_prompt = prompt_template or self._load_prompt("summarize_user")
        content = user_prompt.replace("{{text}}", text[:self.MAX_INPUT_CHARS])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]

        # Kimi 模型使用固定 temperature=1.0
        if self.provider == "kimi" and self.model.startswith("kimi"):
            raw = self.chat(messages)
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
        mime = self.IMAGE_MIME_TYPES.get(ext, self.DEFAULT_IMAGE_MIME)

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

    def close(self) -> None:
        """关闭会话"""
        self._session.close()

    @staticmethod
    def _load_prompt(name: str) -> str:
        """从 prompts/ 目录加载提示词文件，若不存在则返回内置默认值"""
        path = Config.PROMPT_DIR / f"{name}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        # fallback defaults
        defaults = {
            "summarize": (
                "你是一个技术文档分析专家。请对以下内容进行结构化总结，格式如下：\n\n"
                "Summary: [用中文提供简洁的技术总结，200-300字]\n\n"
                "Keywords: [关键词1, 关键词2, 关键词3]\n\n"
                "请确保总结准确、专业，突出技术要点。"
            ),
            "summarize_user": "请总结以下技术文档内容：\n\n{{text}}",
            "structural_summarize": (
                "分析以下技术文档，提取结构化元数据。文档内容：\n\n"
                "{{text}}\n\n"
                "请以 JSON 格式返回（不要包含 markdown 代码块）：\n"
                '{"keywords": ["关键词1", "关键词2"], "entities": [{"name": "实体名", "type": "component", "definition": "一句话定义"}], '
                '"relationships": [{"from": "实体A", "to": "实体B", "relation": "contains", "detail": "关系描述"}], '
                '"answered_questions": ["文档回答了什么问题1", "回答了什么问题2"]}\n\n'
                "要求：\n"
                "- keywords: 5-8 个最重要的技术关键词\n"
                "- entities: 文档中的核心技术实体，每个包含 name、type、definition\n"
                "- relationships: 实体间的关键关系\n"
                "- answered_questions: 文档能够回答的 3-5 个核心问题\n"
                "- 所有内容用中文"
            ),
        }
        return defaults.get(name, "")


class LLMChatClient(BaseLLMClient):
    """LLM 对话专用客户端 - 使用独立的 LLM 配置"""

    def chat_with_thinking(self, messages: List[dict], temperature: float = 0.3) -> str:
        """启用思考模式的对话"""
        # Kimi 模型不支持思考模式，回退到普通对话
        if self.provider == "kimi" and self.model.startswith("kimi"):
            logger.warning("Kimi 模型不支持思考模式，回退到普通对话")
            return self.chat(messages)
        return self.chat(messages, temperature, extra_body={"thinking": {"type": "enabled", "budget": self.THINKING_BUDGET}})

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
