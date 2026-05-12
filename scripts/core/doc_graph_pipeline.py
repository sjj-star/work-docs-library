"""文档图谱处理管道（Doc Graph Pipeline）.

目标：产品文档 → 结构化文本提取 → 实体/关系提取 → 图谱构建 → 持久化.

核心流程：
  1. file-extract 提取文本和表格（主路径）
  2. pdf_parser 提取图片（补充路径）
  3. 章节结构识别
  4. LLM 实体提取（按章节分块）
  5. 规则化表格解析（寄存器表等）
  6. 构建 NetworkX 图谱
  7. 图谱持久化（JSON）
  8. Embedding 向量化（保留，作为 Graph RAG 补充）
"""

import base64
import hashlib
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

import fitz
from parsers.pdf_parser import PDFParser
from PIL import Image

from .batch_clients import BatchClient
from .bigmodel_parser_client import BigModelParserClient
from .config import Config
from .db import KnowledgeDB
from .embedding_client import EmbeddingClient
from .enums import DocumentStatus
from .graph_store import (
    ALL_NODE_TYPES,
    ALL_REL_TYPES,
    GraphEntity,
    GraphRelation,
    GraphStore,
    NetworkXGraphStore,
)
from .models import Chunk, Document
from .vector_index import VectorIndex

logger = logging.getLogger(__name__)

# 常见芯片产品型号正则（可扩展）
_PRODUCT_NAME_PATTERNS = [
    r"TMS320[A-Z0-9]+",
    r"STM32[A-Z0-9]+",
    r"MSP430[A-Z0-9]+",
    r"PIC\d+[A-Z0-9]*",
    r"ATmega[A-Z0-9]+",
    r"EFM32[A-Z0-9]+",
    r"CC\d+[A-Z0-9]*",
    r"DRV\d+[A-Z0-9]*",
]


def _extract_product_name(markdown_text: str, file_path: str) -> str | None:
    """从产品文档中提取产品型号.

    策略（按优先级）：
    1. Markdown 标题/正文前 50 行中匹配型号格式
    2. 文件名中匹配型号格式

    """
    # 1. 从 Markdown 文本提取
    for line in markdown_text.split("\n")[:50]:
        for pattern in _PRODUCT_NAME_PATTERNS:
            match = re.search(pattern, line)
            if match:
                return match.group(0)

    # 2. 从文件名提取
    file_name = Path(file_path).stem
    for pattern in _PRODUCT_NAME_PATTERNS:
        match = re.search(pattern, file_name)
        if match:
            return match.group(0)

    return None


# ---------------------------------------------------------------------------
# 章节解析器：从 file-extract 输出中提取章节结构
# ---------------------------------------------------------------------------


class ChapterNode:
    """章节树节点."""

    def __init__(self, level: int, title: str, content: str = "") -> None:
        """初始化 ChapterNode."""
        self.level = level
        self.title = title.strip()
        self.content = content.strip()
        self.children: list[ChapterNode] = []

    def to_dict(self) -> dict[str, Any]:
        """to_dict 函数."""
        return {
            "level": self.level,
            "title": self.title,
            "content": self.content,
            "children": [c.to_dict() for c in self.children],
        }


class ChapterParser:
    """解析 file-extract 返回的文本，提取章节层级结构（支持 # / ## 树形解析）."""

    @classmethod
    def parse_flat(cls, text: str) -> list[dict[str, Any]]:
        """扁平解析：提取所有标题层级.

        Returns:
            [{"title": str, "level": int, "content": str}, ...]

        """
        lines = text.splitlines()
        chapters: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        in_code_block = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            # 检测 fenced code block 围栏（``` 或 ~~~）
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_block = not in_code_block
                if current is not None:
                    current["_lines"].append(line)
                continue

            if in_code_block:
                if current is not None:
                    current["_lines"].append(line)
                continue

            heading_match = cls._is_heading(line)
            if heading_match:
                if current:
                    current["content"] = "\n".join(current["_lines"]).strip()
                    del current["_lines"]
                    chapters.append(current)

                level, title = heading_match
                current = {"title": title.strip(), "level": level, "_lines": []}
            elif current is not None:
                current["_lines"].append(line)

        if current:
            current["content"] = "\n".join(current["_lines"]).strip()
            del current["_lines"]
            chapters.append(current)

        return chapters

    @classmethod
    def parse_tree(cls, text: str) -> list[ChapterNode]:
        """树形解析：构建真正的多级章节树.

        规则：
        - 按 heading level 确定父子关系（level 小的为父，同级为兄弟）
        - 每个节点保留自己的原始 content，不互相合并
        - 没有 # 时，第一个 heading 自动上移为 #

        Returns:
            List[ChapterNode] — 顶层 # 章节列表

        """
        flat = cls.parse_flat(text)
        if not flat:
            return [ChapterNode(level=1, title="", content=text.strip())]

        # 构建多级树（栈结构）
        roots: list[ChapterNode] = []
        stack: list[ChapterNode] = []

        for ch in flat:
            level = ch["level"]
            node = ChapterNode(level=level, title=ch["title"], content=ch.get("content", ""))

            # 弹出栈中 level >= 当前 level 的节点，找到正确的父节点
            while stack and stack[-1].level >= level:
                stack.pop()

            if stack:
                parent = stack[-1]
                parent.children.append(node)
            else:
                roots.append(node)

            stack.append(node)

        # 如果没有 root（理论上不会发生），将第一个节点作为 root
        if not roots:
            first = flat[0]
            roots = [ChapterNode(level=1, title=first["title"], content=first.get("content", ""))]

        return roots

    @classmethod
    def collect_all_nodes(
        cls, node: ChapterNode, ancestors: list[str] | None = None
    ) -> list[dict[str, str]]:
        """递归收集树中所有有 content 的节点，保留完整标题路径.

        Returns:
            [{"title": str, "content": str}, ...]
        """
        ancestors = ancestors or []
        path = ancestors + [node.title]
        result: list[dict[str, str]] = []
        if node.content:
            path_lines = [f"{'#' * (i + 1)} {t}" for i, t in enumerate(path) if t]
            path_prefix = "\n".join(path_lines)
            full_content = f"{path_prefix}\n\n{node.content}" if path_prefix else node.content
            result.append({"title": node.title, "content": full_content})
        for child in node.children:
            result.extend(cls.collect_all_nodes(child, path))
        return result

    @classmethod
    def _is_heading(cls, line: str) -> tuple[int, str] | None:
        """判断一行是否为 Markdown 标题，返回 (level, title) 或 None.

        仅识别以 # 开头的正规 Markdown 标题，避免将目录条目、代码注释等误识别为标题。
        """
        if not line.startswith("#"):
            return None

        level = 0
        for c in line:
            if c == "#":
                level += 1
            else:
                break

        # Markdown 规范要求 # 序列后必须紧跟至少一个空格
        if level >= len(line) or line[level] != " ":
            return None

        title = line[level + 1 :].strip()
        if not title:
            return None

        # 拒绝纯数字、日期等噪声标题（页码/封面日期误识别）
        stripped = title.strip()
        if re.match(r"^\d+$", stripped):
            return None
        if re.match(
            r"^\d{1,2}\s+"
            r"(January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+\d{4}$",
            stripped,
            re.I,
        ):
            return None
        if re.match(r"^\d{4}-\d{2}-\d{2}$", stripped):
            return None
        return (level, title)


# ---------------------------------------------------------------------------
# Batch 构建器：按 Markdown 层级控制 batch 粒度
# ---------------------------------------------------------------------------


def split_text_by_paragraphs(text: str, max_len: int) -> list[str]:
    """按句子/段落边界切分文本，保护结构化块不被截断.

    提取为模块级函数，供 LLM Batch 和 Embedding Batch 共用。
    """
    blocks: list[str] = []

    def _protect(pattern: re.Pattern, t: str) -> str:
        matches = [m for m in pattern.finditer(t) if m.group(0)]
        if not matches:
            return t
        result = t
        for m in reversed(matches):
            idx = len(blocks)
            blocks.append(m.group(0))
            result = result[: m.start()] + f"\x00BLOCK{idx}\x00" + result[m.end() :]
        return result

    protected = _protect(re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~", re.DOTALL), text)
    protected = _protect(
        re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE),
        protected,
    )
    protected = _protect(
        re.compile(r"(?:^[ \t]*\|.*(?:\r?\n|$))+", re.MULTILINE),
        protected,
    )

    sentences = re.split(r"\n\n+", protected)
    chunks: list[str] = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        for i, block in enumerate(blocks):
            s = s.replace(f"\x00BLOCK{i}\x00", block)
        if len(current) + len(s) + 2 > max_len:
            if current:
                chunks.append(current)
            if len(s) > max_len:
                chunks.append(s)
                current = ""
            else:
                current = s
        else:
            current = (current + "\n\n" + s).strip() if current else s
    if current:
        chunks.append(current)

    if not chunks:
        return [text]
    return chunks


def _split_for_embedding(text: str, max_chars: int) -> list[str]:
    """按语义边界切分文本，确保每段不超过 max_chars.

    切分策略（按优先级）：
    1. 段落边界（保护代码块、HTML table、Markdown table）
    2. 句子边界（标点 [.!?。！？] 后空格）
    3. 若句子仍超长，记录 warning 并保留原样（由 API 错误处理）
    """
    # 第一步：段落切分
    parts = split_text_by_paragraphs(text, max_chars * 2)

    result: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            result.append(part)
            continue

        # 第二步：段落仍超长，按句子边界切分
        sentences = re.split(r"(?<=[.!?。！？])\s+", part)
        current = ""
        current_chars = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            s_chars = len(s)
            if current_chars + s_chars > max_chars and current:
                result.append(current)
                current = s
                current_chars = s_chars
            else:
                current = (current + " " + s).strip() if current else s
                current_chars += s_chars

        if current:
            if len(current) > max_chars:
                logger.warning(
                    f"Embedding 文本仍超长，保留原样 | chars={len(current)} | max_chars={max_chars}"
                )
            result.append(current)

    return result if result else [text]


class BatchBuilder:
    """按 # / ## 层级构建 LLM batch 请求."""

    @classmethod
    def build_batches(
        cls,
        chapters: list[ChapterNode],
        max_chars: int,
    ) -> list[list[dict[str, str]]]:
        """构建 batch 请求列表.

        输入为扁平化的 ChapterNode 列表（每个节点已包含完整标题路径前缀），
        按 max_chars 切分，超长的 content 按段落边界切分为 sub-batch。

        Args:
            chapters: 扁平化的 ChapterNode 列表（无 children）
            max_chars: 每批最大字符数

        Returns:
            每个内部列表代表一个 batch，包含一个或多个 chunk dict

        """
        batches: list[list[dict[str, str]]] = []

        for node in chapters:
            content = node.content
            if not content:
                continue
            chunks = cls._split_if_needed(node.title, content, max_chars)
            for chunk_content in chunks:
                batches.append([{"title": node.title, "content": chunk_content}])

        return batches

    @classmethod
    def _split_if_needed(cls, title: str, content: str, max_len: int) -> list[str]:
        """如果内容超过 max_len，按句子切分."""
        if len(content) <= max_len:
            return [content]
        return cls._split_by_sentences(content, max_len)

    @classmethod
    def _split_by_sentences(cls, text: str, max_len: int) -> list[str]:
        """按句子/段落边界切分文本，保护结构化块不被截断.

        委托给模块级 `split_text_by_paragraphs` 以统一逻辑。
        """
        return split_text_by_paragraphs(text, max_len)


# ---------------------------------------------------------------------------
# LLM 实体提取器
# ---------------------------------------------------------------------------


class EntityExtractor:
    """基于 LLM Batch API 的文档实体和关系提取器."""

    def __init__(self, batch_client=None):
        """初始化 EntityExtractor."""
        self.batch_client = batch_client
        self._system_prompt = self._load_prompt("entity_extraction_system")
        self._user_template = self._load_prompt("entity_extraction_user")

    @staticmethod
    def _load_prompt(name):
        path = Config.PROMPT_DIR / f"{name}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        logger.warning(f"Prompt 文件不存在 | path={path}")
        return ""

    @staticmethod
    def _compress_image_to_base64(img_path: Path) -> str:
        """压缩图片并转为 base64 data URL."""
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                max_edge = Config.LLM_VISION_MAX_EDGE
                quality = Config.LLM_VISION_QUALITY
                if max(w, h) > max_edge:
                    ratio = max_edge / max(w, h)
                    img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                b64 = base64.b64encode(buf.getvalue()).decode()
                return f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            logger.warning(f"图片压缩失败 | path={img_path} | error={e}")
            return ""

    @staticmethod
    def _build_multimodal_content(
        text: str,
        image_base_dir: Path,
        id_prefix: str = "IMG",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """流式解析 Markdown 文本中的图片引用，构建 multimodal content 数组。.

        图片是文档流的一部分，与文字统一处理。遇到 ![alt](path) 即提取路径
        转 base64 插入到对应位置。

        Returns:
            (content_array, image_meta_list)
            content_array 可直接用于 messages[*].content
            image_meta_list 记录每张图片的 image_id 和 rel_path

        """
        pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
        content: list[dict[str, Any]] = []
        image_meta: list[dict[str, Any]] = []
        last_end = 0
        global_idx = 0

        for m in pattern.finditer(text):
            # 1. 图片引用前的文本片段
            if m.start() > last_end:
                content.append({"type": "text", "text": text[last_end : m.start()]})

            # 2. 提取 rel_path，查找实际文件
            rel_path = m.group(2).strip()
            img_path = image_base_dir / rel_path

            # 3. image_id 直接使用 Markdown 引用中 [] 的 alt 文本
            alt_text = m.group(1).strip()
            img_id = alt_text if alt_text else f"{id_prefix}_{global_idx}"
            content.append({"type": "text", "text": f"\n[image_id: {img_id}]\n"})

            if img_path.exists():
                data_url = EntityExtractor._compress_image_to_base64(img_path)
                if data_url:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    )
            else:
                logger.warning(f"图片文件不存在 | rel_path={rel_path} | base_dir={image_base_dir}")

            image_meta.append({"image_id": img_id, "rel_path": rel_path})
            global_idx += 1
            last_end = m.end()

        # 最后一段文本
        if last_end < len(text):
            content.append({"type": "text", "text": text[last_end:]})

        return content, image_meta

    def _build_batch_requests(self, batches, image_base_dir, doc_context=""):
        requests = []
        # 将 user template 按 {{chapters}} 拆分为前缀/后缀，避免章节内容重复
        template = self._user_template
        if "{{chapters}}" in template:
            prefix_tpl, suffix_tpl = template.split("{{chapters}}", 1)
        else:
            prefix_tpl = template
            suffix_tpl = ""
        for i, batch in enumerate(batches):
            chapter_text = "\n\n---\n\n".join(
                f"## {ch['title']}\n\n{ch['content']}" for ch in batch
            )
            prefix = prefix_tpl.replace("{{doc_context}}", doc_context)
            suffix = suffix_tpl

            # 流式解析 Markdown 图片引用，构建 multimodal content
            mm_content, image_meta = self._build_multimodal_content(
                chapter_text, image_base_dir, id_prefix=f"batch_{i}"
            )

            # 组装 content 数组：前缀 + 章节内容(含图片) + 后缀
            content: list[dict[str, Any]] = [{"type": "text", "text": prefix}]
            content.extend(mm_content)
            if suffix:
                content.append({"type": "text", "text": suffix})

            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": content},
            ]
            body = {
                "model": Config.LLM_MODEL,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }
            # Kimi K2.6 等模型 thinking 默认开启，必须显式传递以可控
            thinking_type = "enabled" if Config.LLM_THINKING_ENABLED else "disabled"
            body["extra_body"] = {"thinking": {"type": thinking_type}}
            requests.append(
                {
                    "custom_id": f"batch_{i}",
                    "method": "POST",
                    "url": Config.LLM_BATCH_ENDPOINT,
                    "body": body,
                }
            )
            logger.info(f"Batch {i} 构建完成 | images={len(image_meta)}")
        return requests

    def extract_from_batches(self, batches, image_base_dir=None, doc_id="", doc_context=""):
        """从 batches 构建 requests 并提交 Batch API 提取实体."""
        if not self.batch_client or not batches:
            return [], [], []
        image_base_dir = image_base_dir or Path()
        requests = self._build_batch_requests(batches, image_base_dir, doc_context)
        return self.extract_from_requests(requests, batches, doc_id)

    def extract_from_requests(self, requests, batches, doc_id=""):
        """从已构建的 requests 列表提交 Batch API 提取实体（不重新构建 requests）.

        注意：此方法保留向后兼容，新代码推荐使用 extract_from_results_file.
        """
        if not self.batch_client or not requests:
            return [], [], []

        logger.info(f"提交 Batch 请求 | requests={len(requests)}")
        try:
            results = self.batch_client.submit_parallel_batches(requests)
        except Exception as e:
            logger.error(f"Batch 请求失败 | error={e}")
            return [], [], []

        return self._parse_results(results, batches, doc_id)

    def _parse_results(self, results, batches, doc_id=""):
        """从 Batch API 返回的 results 解析实体/关系/图片描述."""
        all_entities, all_relations, all_image_descriptions = [], [], []
        for i, result in enumerate(results):
            body = result.get("response", {}).get("body", {})
            choices = body.get("choices", [])
            if not choices:
                continue
            content = choices[0].get("message", {}).get("content", "")
            data = self._safe_parse_json(content)
            if not data:
                continue
            batch = batches[i] if i < len(batches) else []
            chapter_title = batch[0]["title"] if batch else ""
            for e in data.get("entities", []):
                if e.get("type") in ALL_NODE_TYPES and e.get("name"):
                    all_entities.append(
                        GraphEntity(
                            entity_type=e["type"],
                            name=e["name"],
                            properties=e.get("properties", {}),
                            source_doc_ids={doc_id},
                            source_chapter=chapter_title,
                        )
                    )
            for r in data.get("relationships", []):
                if r.get("type") in ALL_REL_TYPES:
                    all_relations.append(
                        GraphRelation(
                            rel_type=r["type"],
                            from_name=r.get("from", ""),
                            to_name=r.get("to", ""),
                            from_type=r.get("from_type", ""),
                            to_type=r.get("to_type", ""),
                            properties=r.get("properties", {}),
                            source_doc_ids={doc_id},
                        )
                    )
            for img_desc in data.get("image_descriptions", []):
                all_image_descriptions.append(img_desc)

        logger.info(f"实体提取完成 | entities={len(all_entities)} | relations={len(all_relations)}")
        return all_entities, all_relations, all_image_descriptions

    def extract_from_results_file(
        self,
        results_path: Path,
        doc_id: str = "",
        chapter_map: dict[str, str] | None = None,
    ) -> tuple[list[GraphEntity], list[GraphRelation], list[dict[str, Any]]]:
        """从磁盘上的 results.jsonl 文件解析实体/关系.

        Args:
            results_path: Batch API 结果文件路径（JSONL 格式）
            doc_id: 文档 ID
            chapter_map: custom_id -> chapter_title 映射（可选）

        Returns:
            (entities, relations, image_descriptions)
        """
        if not results_path.exists():
            logger.warning(f"结果文件不存在 | path={results_path}")
            return [], [], []

        from core.batch_clients import _parse_jsonl

        results = _parse_jsonl(results_path.read_text(encoding="utf-8"))
        logger.info(f"从文件解析 Batch 结果 | path={results_path} | results={len(results)}")

        all_entities, all_relations, all_image_descriptions = [], [], []
        for result in results:
            custom_id = result.get("custom_id", "")
            body = result.get("response", {}).get("body", {})
            choices = body.get("choices", [])
            if not choices:
                continue
            content = choices[0].get("message", {}).get("content", "")
            data = self._safe_parse_json(content)
            if not data:
                continue

            chapter_title = chapter_map.get(custom_id, "") if chapter_map else ""
            for e in data.get("entities", []):
                if e.get("type") in ALL_NODE_TYPES and e.get("name"):
                    all_entities.append(
                        GraphEntity(
                            entity_type=e["type"],
                            name=e["name"],
                            properties=e.get("properties", {}),
                            source_doc_ids={doc_id},
                            source_chapter=chapter_title,
                        )
                    )
            for r in data.get("relationships", []):
                if r.get("type") in ALL_REL_TYPES:
                    all_relations.append(
                        GraphRelation(
                            rel_type=r["type"],
                            from_name=r.get("from", ""),
                            to_name=r.get("to", ""),
                            from_type=r.get("from_type", ""),
                            to_type=r.get("to_type", ""),
                            properties=r.get("properties", {}),
                            source_doc_ids={doc_id},
                        )
                    )
            for img_desc in data.get("image_descriptions", []):
                all_image_descriptions.append(img_desc)

        logger.info(f"文件解析完成 | entities={len(all_entities)} | relations={len(all_relations)}")
        return all_entities, all_relations, all_image_descriptions

    def _safe_parse_json(self, raw):
        if not raw:
            return None
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning(f"JSON 解析失败 | raw={raw[:200]}...")
                return None


class DocGraphPipeline:
    """文档图谱处理管道.

    产品文档 → file-extract → 章节解析 → 实体提取 → 图谱构建 → 持久化.
    """

    def __init__(
        self,
        db: KnowledgeDB | None = None,
        vec: VectorIndex | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        """初始化 DocGraphPipeline."""
        self.db = db or KnowledgeDB()
        self.vec = vec or VectorIndex(dim=Config.EMBEDDING_DIMENSION)
        self.graph = graph_store or NetworkXGraphStore()
        self._owns_graph = graph_store is None

        # 客户端
        try:
            self.parser_client = BigModelParserClient()
            logger.info("BigModelParserClient 已初始化")
        except RuntimeError as e:
            logger.warning(f"BigModelParserClient 初始化失败: {e}")
            self.parser_client = None

        try:
            self.llm_batch = BatchClient()
            logger.info("LLM BatchClient 已初始化")
        except RuntimeError as e:
            logger.warning(f"LLM BatchClient 初始化失败: {e}")
            self.llm_batch = None

        # Embedding 已改为同步 API，不再使用 BatchClient
        self.embed_batch = None

        # 解析器（仅 PDF，Office 文件暂时不支持）
        self.parsers = {
            ".pdf": PDFParser(),
        }

        # 子组件
        self.chapter_parser = ChapterParser()
        self.entity_extractor = EntityExtractor(self.llm_batch)
        self.embedding_batch_client = None

    def scan(self, path: str) -> list[str]:
        """扫描文件."""
        p = Path(path)
        files = []
        if p.is_file():
            files.append(str(p.resolve()))
        elif p.is_dir():
            for ext in self.parsers.keys():
                files.extend(str(f) for f in p.rglob(f"*{ext}"))
        return sorted(set(files))

    def ingest(
        self,
        path: str,
        dry_run: bool = False,
        force: bool = False,
    ) -> list[str]:
        """主入口：处理文档集合并构建图谱.

        Returns:
            处理成功的 doc_id 列表

        """
        files = self.scan(path)
        ingested: list[str] = []

        for f in files:
            doc_id = self._process_one(f, dry_run=dry_run, force=force)
            if doc_id:
                ingested.append(doc_id)

        return ingested

    # ------------------------------------------------------------------
    # 三阶段拆分方法
    # ------------------------------------------------------------------

    def stage1_parse(self, file_path: str) -> tuple[str, Path, str, list[Path]]:
        """阶段1: PDF → Markdown + images.

        Returns:
            (doc_id, parsed_output_dir, extracted_text, image_paths)
        """
        suffix = Path(file_path).suffix.lower()
        parser = self.parsers.get(suffix)
        if not parser:
            raise ValueError(f"不支持的文件格式 | file={file_path}")

        file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
        doc_id = file_hash[:16]
        parsed_output_dir = Config.DB_PATH.parent / "parsed" / doc_id
        parsed_output_dir.mkdir(parents=True, exist_ok=True)

        extracted_text = ""
        bigmodel_images: list[Path] = []

        # BigModel Expert 解析（主路径）
        if self.parser_client and suffix == ".pdf":
            try:
                extracted_text, bigmodel_images = self.parser_client.parse_pdf(
                    file_path,
                    output_dir=parsed_output_dir,
                    tool_type="expert",
                )
                logger.info(
                    f"BigModel 解析成功 | text={len(extracted_text)} chars | "
                    f"images={len(bigmodel_images)}"
                )
            except Exception as e:
                logger.error(f"BigModel 解析失败 | error={e}，回退到本地解析")

        # 回退到本地解析
        if not extracted_text:
            extracted_text, local_images = parser.parse(file_path, output_dir=parsed_output_dir)
            bigmodel_images = local_images
            logger.info(
                f"本地解析回退 | length={len(extracted_text)} chars | images={len(local_images)}"
            )

        # 保存 result.md（与 BigModel 输出格式一致）
        result_md_path = parsed_output_dir / "result.md"
        result_md_path.write_text(extracted_text, encoding="utf-8")
        logger.info(f"result.md 已保存 | path={result_md_path} | chars={len(extracted_text)}")

        return doc_id, parsed_output_dir, extracted_text, bigmodel_images

    def stage2_build_jsonl(
        self,
        doc_id: str,
        max_chars: int | None = None,
    ) -> tuple[Path, list[list[dict[str, str]]], list[dict[str, Any]]]:
        """阶段2: Markdown → Batch JSONL.

        从 knowledge_base/parsed/{doc_id}/result.md 读取，
        生成 JSONL 到 knowledge_base/batch/{doc_id}.jsonl。

        Returns:
            (jsonl_path, batches, requests)
        """
        parsed_output_dir = Config.DB_PATH.parent / "parsed" / doc_id
        result_md_path = parsed_output_dir / "result.md"
        if not result_md_path.exists():
            raise FileNotFoundError(f"result.md 不存在 | path={result_md_path}")

        extracted_text = result_md_path.read_text(encoding="utf-8")
        tree_chapters = self.chapter_parser.parse_tree(extracted_text)
        logger.info(f"章节解析完成 | roots={len(tree_chapters)}")

        new_chapters_flat: list[dict[str, Any]] = []
        for root in tree_chapters:
            new_chapters_flat.extend(ChapterParser.collect_all_nodes(root))

        # 构建 batch（全部章节，不做增量过滤）
        process_nodes = [
            ChapterNode(level=1, title=ch["title"], content=ch["content"])
            for ch in new_chapters_flat
        ]
        max_chars = max_chars or Config.LLM_BATCH_MAX_CHARS
        batches = BatchBuilder.build_batches(process_nodes, max_chars) if process_nodes else []
        logger.info(f"Batch 构建完成 | batches={len(batches)}")

        # 构建 doc_context
        doc_title = tree_chapters[0].title if tree_chapters else ""
        product_name = _extract_product_name(extracted_text, "") or ""
        doc_context_parts = []
        if doc_title:
            doc_context_parts.append(f"文档《{doc_title}》")
        if product_name:
            doc_context_parts.append(f"产品型号 {product_name}")
        doc_context = ""
        if doc_context_parts:
            doc_context = "以下章节来自 " + "，".join(doc_context_parts) + "。\n\n---\n"

        # 构建 requests
        requests = self.entity_extractor._build_batch_requests(
            batches=batches,
            image_base_dir=parsed_output_dir,
            doc_context=doc_context,
        )
        logger.info(f"Requests 构建完成 | requests={len(requests)}")

        # 保存 JSONL
        batch_dir = Config.DB_PATH.parent / "batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = batch_dir / f"{doc_id}.jsonl"
        from core.batch_clients import _build_jsonl

        _build_jsonl(requests, jsonl_path)
        logger.info(f"JSONL 已生成 | path={jsonl_path} | requests={len(requests)}")

        # 保存 batch_info 映射（request index -> chapter titles）
        batch_info = []
        for i, batch in enumerate(batches):
            titles = [ch["title"] for ch in batch]
            batch_info.append(
                {
                    "index": i,
                    "custom_id": f"batch_{i}",
                    "chapter_titles": titles,
                }
            )
        batch_info_path = batch_dir / f"{doc_id}_batch_info.json"
        batch_info_path.write_text(
            json.dumps(batch_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Batch info 已保存 | path={batch_info_path} | batches={len(batch_info)}")

        return jsonl_path, batches, requests

    def _read_result_md(self, doc_id: str) -> str:
        """读取 parsed/{doc_id}/result.md."""
        result_md_path = Config.DB_PATH.parent / "parsed" / doc_id / "result.md"
        if not result_md_path.exists():
            raise FileNotFoundError(f"result.md 不存在 | path={result_md_path}")
        return result_md_path.read_text(encoding="utf-8")

    def _incremental_analysis(
        self,
        file_path: str,
        extracted_text: str,
        force: bool = False,
    ) -> tuple[
        list[dict[str, Any]],
        list[tuple[dict[str, Any], list[Chunk]]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[Chunk],
        Document | None,
        str,
    ]:
        """章节级增量分析.

        Returns:
            (new_chapters_flat, unchanged, changed, added, removed, old_doc, file_hash)
        """
        file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()

        # 章节解析
        tree_chapters = self.chapter_parser.parse_tree(extracted_text)
        new_chapters_flat: list[dict[str, Any]] = []
        for root in tree_chapters:
            new_chapters_flat.extend(ChapterParser.collect_all_nodes(root))

        # 与旧数据比较
        old_doc = self.db.get_document_by_path(file_path)
        old_chunks: list[Chunk] = []
        if old_doc:
            old_chunks = self.db.query_by_doc(old_doc.doc_id)

        from collections import defaultdict

        old_chunk_map: dict[str, list[Chunk]] = defaultdict(list)
        for ck in old_chunks:
            old_chunk_map[ck.chapter_title].append(ck)
        unchanged: list[tuple[dict[str, Any], list[Chunk]]] = []
        changed: list[dict[str, Any]] = []
        added: list[dict[str, Any]] = []
        removed: list[Chunk] = []

        for ch in new_chapters_flat:
            ch_hash = hashlib.md5(ch["content"].encode()).hexdigest()[:16]
            ch["content_hash"] = ch_hash
            if ch["title"] in old_chunk_map:
                old_cks = old_chunk_map[ch["title"]]
                # sub-chunks 共享 content_hash，只要有一个匹配即认为整个 chapter 未变
                if any(
                    old_ck.metadata.get("content_hash", "") == ch_hash
                    for old_ck in old_cks
                ) and not force:
                    unchanged.append((ch, old_cks))
                else:
                    changed.append(ch)
            else:
                added.append(ch)

        for title, old_cks in old_chunk_map.items():
            if title not in {c["title"] for c in new_chapters_flat}:
                removed.extend(old_cks)

        logger.info(
            f"章节增量分析 | 未变={len(unchanged)} | "
            f"变更={len(changed)} | 新增={len(added)} | 删除={len(removed)}"
        )
        return new_chapters_flat, unchanged, changed, added, removed, old_doc, file_hash

    @staticmethod
    def _is_valid_results_file(path: Path) -> bool:
        """校验 results.jsonl 文件是否包含至少一个有效的 JSON 行."""
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            with open(path, encoding="utf-8") as f:
                first_line = f.readline().strip()
                if not first_line:
                    return False
                json.loads(first_line)
                return True
        except (json.JSONDecodeError, OSError):
            return False

    def stage3_submit_batches(
        self,
        doc_id: str,
        file_path: str,
        jsonl_path: Path | None = None,
        force: bool = False,
    ) -> Path:
        """阶段3: 提交 Batch API 并保存原始结果.

        Returns:
            results_path: knowledge_base/batch/{doc_id}_results.jsonl
        """
        batch_dir = Config.DB_PATH.parent / "batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        results_path = batch_dir / f"{doc_id}_results.jsonl"

        # 检查是否需要处理
        file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
        if not force:
            existing = self.db.get_document_by_path(file_path)
            if existing and existing.file_hash == file_hash:
                if existing.status == DocumentStatus.DONE:
                    logger.info(f"跳过未变更文档 | file={file_path} | doc_id={doc_id}")
                    if self._is_valid_results_file(results_path):
                        return results_path
                if (
                    existing.status == DocumentStatus.BATCH_SUBMITTED
                    and self._is_valid_results_file(results_path)
                ):
                    logger.info(
                        f"Batch 已提交且结果文件有效，跳过 | file={file_path} | doc_id={doc_id}"
                    )
                    return results_path
            # 如果结果文件已存在且有效且非 force，直接返回
            if self._is_valid_results_file(results_path):
                logger.info(f"结果文件已存在且有效，跳过提交 | path={results_path}")
                return results_path

        logger.info(f"开始阶段3 Batch 提交 | file={file_path} | doc_id={doc_id}")
        self.db.update_document_status(doc_id, DocumentStatus.PROCESSING)

        extracted_text = self._read_result_md(doc_id)
        new_chapters_flat, unchanged, changed, added, removed, old_doc, _ = (
            self._incremental_analysis(file_path, extracted_text, force)
        )

        # 尝试从持久化 JSONL 读取 requests（支持用户编辑后重新提交）
        requests: list[dict[str, Any]] = []
        if jsonl_path and jsonl_path.exists() and jsonl_path.stat().st_size > 0:
            from core.batch_clients import _parse_jsonl

            all_requests = _parse_jsonl(jsonl_path.read_text(encoding="utf-8"))
            # 通过 batch_info 做增量过滤：只保留对应变更/新增章节的 requests
            batch_info_path = jsonl_path.parent / f"{doc_id}_batch_info.json"
            if batch_info_path.exists() and (changed or added):
                batch_info = json.loads(batch_info_path.read_text(encoding="utf-8"))
                changed_titles = {ch["title"] for ch in changed + added}
                changed_custom_ids: set[str] = set()
                for info in batch_info:
                    titles = set(info.get("chapter_titles", []))
                    if titles & changed_titles:
                        changed_custom_ids.add(info.get("custom_id", ""))
                requests = [
                    req for req in all_requests if req.get("custom_id") in changed_custom_ids
                ]
                logger.info(
                    f"从 JSONL 增量过滤 | path={jsonl_path} | "
                    f"total={len(all_requests)} | filtered={len(requests)}"
                )
            else:
                requests = all_requests
                logger.info(
                    f"从 JSONL 读取全部 requests | path={jsonl_path} | count={len(requests)}"
                )

            # 确保每个 request 的 body 中都有 thinking 参数（如用户编辑删除了则补充）
            thinking_type = "enabled" if Config.LLM_THINKING_ENABLED else "disabled"
            for req in requests:
                body = req.get("body", {})
                if "extra_body" not in body:
                    body["extra_body"] = {"thinking": {"type": thinking_type}}
                elif "thinking" not in body.get("extra_body", {}):
                    body.setdefault("extra_body", {})["thinking"] = {"type": thinking_type}

            if not self.llm_batch:
                logger.info("BatchClient 不可用，跳过 API 提交")
                results_path.write_text("", encoding="utf-8")
                return results_path

        # 回退：重新构建 requests（JSONL 不存在或为空时）
        if not requests:
            process_nodes: list[ChapterNode] = []
            for ch in changed + added:
                process_nodes.append(ChapterNode(level=1, title=ch["title"], content=ch["content"]))
            batches = (
                BatchBuilder.build_batches(process_nodes, Config.LLM_BATCH_MAX_CHARS)
                if process_nodes
                else []
            )

            if not batches or not self.llm_batch:
                logger.info("无变更章节或 BatchClient 不可用，跳过 API 提交")
                results_path.write_text("", encoding="utf-8")
                return results_path

            parsed_output_dir = Config.DB_PATH.parent / "parsed" / doc_id
            doc_title = ""
            tree_chapters = self.chapter_parser.parse_tree(extracted_text)
            if tree_chapters:
                doc_title = tree_chapters[0].title
            product_name = _extract_product_name(extracted_text, file_path) or ""
            doc_context_parts = []
            if doc_title:
                doc_context_parts.append(f"文档《{doc_title}》")
            if product_name:
                doc_context_parts.append(f"产品型号 {product_name}")
            doc_context = ""
            if doc_context_parts:
                doc_context = "以下章节来自 " + "，".join(doc_context_parts) + "。\n\n---\n"

            requests = self.entity_extractor._build_batch_requests(
                batches=batches,
                image_base_dir=parsed_output_dir,
                doc_context=doc_context,
            )

        # 删除旧结果文件（如果存在）
        if results_path.exists():
            results_path.unlink()

        # 提交并保存结果
        logger.info(f"提交 Batch | requests={len(requests)} | output={results_path}")
        results = self.llm_batch.submit_parallel_batches(requests, output_path=results_path)
        # fallback：如果 BatchClient 未写入文件（如 Mock 客户端），手动写入
        if results and (not results_path.exists() or results_path.stat().st_size == 0):
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Batch 结果已保存 | path={results_path}")
        # 保存增量分析结果摘要，供 stage4 校验一致性
        incremental_info = {
            "unchanged_titles": [ch["title"] for ch, _ in unchanged],
            "changed_titles": [ch["title"] for ch in changed],
            "added_titles": [ch["title"] for ch in added],
            "removed_titles": [ck.chapter_title for ck in removed],
            "result_md_hash": hashlib.md5(
                (Config.DB_PATH.parent / "parsed" / doc_id / "result.md").read_bytes()
            ).hexdigest(),
        }
        info_path = batch_dir / f"{doc_id}_incremental.json"
        info_path.write_text(
            json.dumps(incremental_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.db.update_document_status(doc_id, DocumentStatus.BATCH_SUBMITTED)
        return results_path

    def stage4_ingest_results(
        self,
        doc_id: str,
        file_path: str,
        results_path: Path,
        force: bool = False,
    ) -> str:
        """阶段4: 从 Batch 结果文件解析并入库.

        Returns:
            doc_id
        """
        # 确保每个文档使用干净的局部图谱（仅清空局部创建的图）
        if self._owns_graph and hasattr(self.graph, "clear"):
            self.graph.clear()

        # 检查是否需要重新处理
        file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
        if not force:
            existing = self.db.get_document_by_path(file_path)
            if (
                existing
                and existing.file_hash == file_hash
                and existing.status == DocumentStatus.DONE
            ):
                logger.info(f"跳过未变更文档 | file={file_path} | doc_id={doc_id}")
                return existing.doc_id

        logger.info(f"开始阶段4入库 | file={file_path} | doc_id={doc_id}")
        self.db.update_document_status(doc_id, DocumentStatus.PROCESSING)

        suffix = Path(file_path).suffix.lower()

        # 获取 PDF 实际页数
        pdf_page_count = 0
        if suffix == ".pdf":
            try:
                with fitz.open(str(file_path)) as doc:
                    pdf_page_count = len(doc)
            except Exception as e:
                logger.warning(f"获取 PDF 页数失败 | error={e}")

        extracted_text = self._read_result_md(doc_id)
        new_chapters_flat, unchanged, changed, added, removed, old_doc, _ = (
            self._incremental_analysis(file_path, extracted_text, force)
        )

        # 校验增量分析结果与 stage3 是否一致
        info_path = results_path.parent / f"{doc_id}_incremental.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            expected_hash = info.get("result_md_hash", "")
            actual_hash = hashlib.md5(
                (Config.DB_PATH.parent / "parsed" / doc_id / "result.md").read_bytes()
            ).hexdigest()
            if expected_hash != actual_hash:
                logger.warning(
                    f"result.md 在 stage3 后被修改，增量分析结果可能不一致 | doc_id={doc_id}"
                )
            else:
                current_titles = {
                    "unchanged": [ch["title"] for ch, _ in unchanged],
                    "changed": [ch["title"] for ch in changed],
                    "added": [ch["title"] for ch in added],
                    "removed": [ck.chapter_title for ck in removed],
                }
                for key in ("unchanged_titles", "changed_titles", "added_titles", "removed_titles"):
                    expected = set(info.get(key, []))
                    actual = set(current_titles.get(key.replace("_titles", ""), []))
                    if expected != actual:
                        logger.warning(f"增量分析结果与 stage3 不一致 ({key}) | doc_id={doc_id}")

        # --- 实体提取 ---
        all_entities: list[GraphEntity] = []
        all_relations: list[GraphRelation] = []
        all_image_descriptions: list[dict[str, Any]] = []

        # 复用未变章节的实体缓存
        for ch, old_cks in unchanged:
            # sub-chunks 共享相同的缓存元数据，从第一个取即可
            old_ck = old_cks[0]
            cached_ents = old_ck.metadata.get("extracted_entities", [])
            cached_rels = old_ck.metadata.get("extracted_relations", [])
            cached_imgs = old_ck.metadata.get("image_descriptions", [])
            for e in cached_ents:
                all_entities.append(
                    GraphEntity(
                        entity_type=e.get("type", ""),
                        name=e.get("name", ""),
                        properties=e.get("properties", {}),
                        source_doc_ids={doc_id},
                        source_chapter=ch["title"],
                    )
                )
            for r in cached_rels:
                all_relations.append(
                    GraphRelation(
                        rel_type=r.get("type", ""),
                        from_name=r.get("from", ""),
                        to_name=r.get("to", ""),
                        from_type=r.get("from_type", ""),
                        to_type=r.get("to_type", ""),
                        properties=r.get("properties", {}),
                        source_doc_ids={doc_id},
                    )
                )
            all_image_descriptions.extend(cached_imgs)

        # 从结果文件解析变更章节
        if results_path.exists() and results_path.stat().st_size > 0:
            # 构建 chapter_map（从 batch_info.json）
            batch_info_path = results_path.parent / f"{doc_id}_batch_info.json"
            chapter_map: dict[str, str] = {}
            if batch_info_path.exists():
                batch_info = json.loads(batch_info_path.read_text(encoding="utf-8"))
                for info in batch_info:
                    titles = info.get("chapter_titles", [])
                    chapter_map[info.get("custom_id", "")] = titles[0] if titles else ""

            ents, rels, img_descs = self.entity_extractor.extract_from_results_file(
                results_path=results_path,
                doc_id=doc_id,
                chapter_map=chapter_map,
            )
            all_entities.extend(ents)
            all_relations.extend(rels)
            all_image_descriptions.extend(img_descs)

        logger.info(
            f"实体提取完成 | entities={len(all_entities)} | "
            f"relations={len(all_relations)} | image_descs={len(all_image_descriptions)}"
        )

        # --- 构建图谱 ---
        entity_map: dict[str, GraphEntity] = {}
        for e in all_entities:
            key = f"{e.entity_type}::{e.name}"
            if key in entity_map:
                entity_map[key].properties.update(e.properties)
            else:
                entity_map[key] = e

        for e in entity_map.values():
            self.graph.add_entity(e)

        for r in all_relations:
            self.graph.add_relation(r)

        # 提取产品型号并建立 Product-Module 关系
        product_name = _extract_product_name(extracted_text, file_path)
        if product_name:
            product_entity = GraphEntity(
                entity_type="Product",
                name=product_name,
                properties={"description": f"芯片产品型号: {product_name}"},
                source_doc_ids={doc_id},
            )
            self.graph.add_entity(product_entity)
            for module_name in {e.name for e in all_entities if e.entity_type == "Module"}:
                self.graph.add_relation(
                    GraphRelation(
                        rel_type="HAS_MODULE",
                        from_name=product_name,
                        from_type="Product",
                        to_name=module_name,
                        to_type="Module",
                        source_doc_ids={doc_id},
                    )
                )
            logger.info(
                f"产品型号提取完成 | product={product_name} | "
                f"modules={len({e.name for e in all_entities if e.entity_type == 'Module'})}"
            )

        # --- 保存 chunks → SQLite → 向量化 ---
        try:
            self._save_chunks_to_db(
                doc_id,
                file_path,
                file_hash,
                new_chapters_flat,
                extracted_text,
                unchanged,
                changed,
                added,
                removed,
                all_image_descriptions,
                all_entities,
                all_relations,
            )
        except Exception:
            self.db.update_document_status(doc_id, DocumentStatus.FAILED)
            raise

        # --- 持久化图谱 ---
        graph_path = Config.DB_PATH.parent / Config.GRAPH_OUTPUT_DIR / f"{doc_id}.json"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph.save(graph_path)
        logger.info(f"图谱已保存 | path={graph_path} | {self.graph.stats()}")

        # 删除旧图谱文件
        if old_doc and old_doc.doc_id != doc_id:
            old_graph_path = (
                Config.DB_PATH.parent / Config.GRAPH_OUTPUT_DIR / f"{old_doc.doc_id}.json"
            )
            if old_graph_path.exists():
                old_graph_path.unlink()
                logger.info(f"已删除旧图谱文件 | {old_graph_path}")

        # --- 更新文档状态 ---
        self.db.upsert_document(
            Document(
                doc_id=doc_id,
                title=Path(file_path).stem,
                source_path=file_path,
                file_type=suffix.lstrip("."),
                total_pages=pdf_page_count,
                file_hash=file_hash,
                status=DocumentStatus.DONE,
            )
        )

        logger.info(f"文档入库完成 | doc_id={doc_id}")
        return doc_id

    def stage5_build_embed_jsonl(self, doc_id: str) -> Path:
        """阶段5: 从 SQLite chunks 构建 Embedding Batch JSONL（本地，不调用 API）.

        Returns:
            embed_jsonl_path: `knowledge_base/batch/{doc_id}_embed.jsonl`
        """
        batch_dir = Config.DB_PATH.parent / "batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        embed_jsonl_path = batch_dir / f"{doc_id}_embed.jsonl"

        db_chunks = self.db.query_by_doc(doc_id)
        reembed_pairs: list[tuple[str, int]] = []

        for ck in db_chunks:
            if not ck.content.strip():
                continue
            if ck.metadata.get("embedding"):
                continue
            reembed_pairs.append((ck.content, ck.id))

        if not reembed_pairs:
            embed_jsonl_path.write_text("", encoding="utf-8")
            logger.info(f"所有 chunk 已有 embedding，生成空 JSONL | path={embed_jsonl_path}")
            return embed_jsonl_path

        # 构建 requests，每个 request 包含单个文本，custom_id 直接编码 db_id
        requests: list[dict[str, Any]] = []
        for text, db_id in reembed_pairs:
            requests.append(
                {
                    "custom_id": f"embed_dbid_{db_id}",
                    "method": "POST",
                    "url": Config.EMBEDDING_BATCH_ENDPOINT,
                    "body": {
                        "model": Config.EMBEDDING_MODEL,
                        "input": text,
                    },
                }
            )

        from core.batch_clients import _build_jsonl

        _build_jsonl(requests, embed_jsonl_path)
        logger.info(
            f"Embedding JSONL 已生成 | path={embed_jsonl_path} | "
            f"chunks={len(reembed_pairs)} | requests={len(requests)}"
        )
        return embed_jsonl_path

    def stage6_submit_embed_batches(
        self,
        doc_id: str,
        embed_jsonl_path: Path | None = None,
    ) -> str:
        """阶段6: 提交 Embedding Batch API 并解析结果入库.

        Args:
            doc_id: 文档 ID
            embed_jsonl_path: Embedding JSONL 路径，默认从 batch 目录读取

        Returns:
            doc_id
        """
        batch_dir = Config.DB_PATH.parent / "batch"
        if embed_jsonl_path is None:
            embed_jsonl_path = batch_dir / f"{doc_id}_embed.jsonl"

        # 1. 收集复用 embedding
        db_chunks = self.db.query_by_doc(doc_id)
        all_items: list[tuple[int, list[float]]] = []
        reembed_db_ids: set[int] = set()

        for ck in db_chunks:
            emb = ck.metadata.get("embedding")
            if emb:
                all_items.append((ck.id, emb))
            else:
                reembed_db_ids.add(ck.id)

        if not reembed_db_ids:
            logger.info(f"所有 chunk 已有 embedding，跳过 stage6 | doc_id={doc_id}")
            for ck in db_chunks:
                self.db.update_chunk_status(ck.id, "done")
            return doc_id

        # 2. 从 JSONL 读取并同步向量化
        if not embed_jsonl_path.exists() or embed_jsonl_path.stat().st_size == 0:
            logger.warning(f"Embedding JSONL 为空，跳过向量化 | path={embed_jsonl_path}")
        else:
            from core.batch_clients import _parse_jsonl

            requests = _parse_jsonl(embed_jsonl_path.read_text(encoding="utf-8"))
            if requests:
                logger.info(f"同步 Embedding | requests={len(requests)} | doc_id={doc_id}")
                embedder = EmbeddingClient()
                for req in requests:
                    custom_id = req.get("custom_id", "")
                    # custom_id 格式: embed_dbid_{db_id}
                    try:
                        db_id = int(custom_id.split("_")[-1])
                    except (ValueError, IndexError):
                        logger.warning(f"无法从 custom_id 解析 db_id | custom_id={custom_id}")
                        continue
                    text = req.get("body", {}).get("input", "")
                    if not text:
                        logger.warning(f"JSONL request 中 input 为空 | custom_id={custom_id}")
                        continue
                    emb = embedder.embed_single(text)
                    all_items.append((db_id, emb))
                embedder.close()

        # 4. 统一写入 SQLite 和 FAISS（带失败回滚）
        if all_items:
            self.db.update_chunks_embedded_batch(all_items)
            try:
                self.vec.add_batch(all_items)
            except Exception as e:
                logger.error(
                    f"FAISS 添加失败，回滚 SQLite embedding | doc_id={doc_id} | error={e}"
                )
                for chunk_db_id, _ in all_items:
                    ck = self.db.get_chunk_by_db_id(chunk_db_id)
                    if ck:
                        meta = ck.metadata.copy()
                        meta.pop("embedding", None)
                        with self.db._connect() as conn:
                            conn.execute(
                                "UPDATE chunks SET metadata = ?, status = 'embedded' WHERE id = ?",
                                (json.dumps(meta, ensure_ascii=False), chunk_db_id),
                            )
                raise

        # 5. 更新状态为 done
        for ck in db_chunks:
            self.db.update_chunk_status(ck.id, "done")

        return doc_id

    def stage3_ingest(
        self,
        file_path: str,
        doc_id: str,
        parsed_output_dir: Path,
        extracted_text: str,
        bigmodel_images: list[Path],
        jsonl_path: Path | None = None,
        force: bool = False,
    ) -> str:
        """阶段3: JSONL → API → 实体提取 → 向量化 → 入库.

        已废弃：内部委托给 stage3_submit_batches + stage4_ingest_results。
        保留此方法以保证向后兼容。
        """
        results_path = self.stage3_submit_batches(
            doc_id=doc_id,
            file_path=file_path,
            jsonl_path=jsonl_path,
            force=force,
        )
        self.stage4_ingest_results(
            doc_id=doc_id,
            file_path=file_path,
            results_path=results_path,
            force=force,
        )
        embed_jsonl_path = self.stage5_build_embed_jsonl(doc_id)
        return self.stage6_submit_embed_batches(doc_id, embed_jsonl_path)

    def _process_one(
        self,
        file_path: str,
        dry_run: bool = False,
        force: bool = False,
    ) -> str | None:
        """处理单个文档（完整流程，兼容入口）."""
        if dry_run:
            file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
            return file_hash[:16]

        file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
        doc_id = file_hash[:16]

        try:
            # 阶段1: PDF → Markdown
            doc_id, _parsed_output_dir, _extracted_text, _bigmodel_images = self.stage1_parse(
                file_path
            )

            # 阶段2: Markdown → JSONL
            jsonl_path, _batches, _requests = self.stage2_build_jsonl(doc_id)

            # 阶段3: 提交 Batch API 并保存结果
            results_path = self.stage3_submit_batches(
                doc_id=doc_id,
                file_path=file_path,
                jsonl_path=jsonl_path,
                force=force,
            )

            # 阶段4: 从结果文件解析并入库
            self.stage4_ingest_results(
                doc_id=doc_id,
                file_path=file_path,
                results_path=results_path,
                force=force,
            )

            # 阶段5: 构建 Embedding Batch JSONL
            embed_jsonl_path = self.stage5_build_embed_jsonl(doc_id)

            # 阶段6: 提交 Embedding Batch API 并解析入库
            return self.stage6_submit_embed_batches(doc_id, embed_jsonl_path)
        except Exception as e:
            logger.error(f"文档处理失败 | file={file_path} | doc_id={doc_id} | error={e}")
            try:
                self.db.update_document_status(doc_id, DocumentStatus.FAILED)
            except Exception:
                pass
            raise

    def _save_chunks_to_db(
        self,
        doc_id: str,
        file_path: str,
        file_hash: str,
        chapters: list[dict[str, Any]],
        full_text: str,
        unchanged: list[tuple[dict[str, Any], list[Chunk]]] | None = None,
        changed: list[dict[str, Any]] | None = None,
        added: list[dict[str, Any]] | None = None,
        removed: list[Chunk] | None = None,
        image_descriptions: list[dict[str, Any]] | None = None,
        all_entities: list[GraphEntity] | None = None,
        all_relations: list[GraphRelation] | None = None,
    ) -> None:
        """将章节作为 chunks 保存到 SQLite（增量模式），支持向量检索，并合并图片文字化结果."""
        unchanged = unchanged or []
        changed = changed or []
        added = added or []
        removed = removed or []
        image_descriptions = image_descriptions or []
        all_entities = all_entities or []
        all_relations = all_relations or []

        # 1. 清理旧数据（所有旧 chunks 统一清理）
        old_doc = self.db.get_document_by_path(file_path)
        if old_doc:
            all_old = self.db.query_by_doc(old_doc.doc_id)
            if all_old:
                self.vec.remove_doc([c.id for c in all_old])
            self.db.delete_chunks_by_doc(old_doc.doc_id)

        # 2. 准备实体/关系缓存序列化
        entity_dicts = [e.to_dict() for e in all_entities]
        relation_dicts = [r.to_dict() for r in all_relations]

        def _build_metadata(ch_title: str, ch_hash: str, old_ck: Chunk | None) -> dict[str, Any]:
            """构建 chunk metadata，包含缓存的实体/关系/embedding."""
            ch_ents = [e for e in entity_dicts if e.get("source_chapter") == ch_title]
            ch_ent_keys = {(e.get("type", ""), e.get("name", "")) for e in ch_ents}
            ch_rels = [
                r
                for r in relation_dicts
                if (r.get("from_type", ""), r.get("from", "")) in ch_ent_keys
                or (r.get("to_type", ""), r.get("to", "")) in ch_ent_keys
            ]
            ch_images = [img for img in image_descriptions if img.get("chapter_title") == ch_title]
            meta: dict[str, Any] = {
                "content_hash": ch_hash,
                "extracted_entities": ch_ents,
                "extracted_relations": ch_rels,
                "image_descriptions": ch_images,
            }
            if old_ck and old_ck.metadata.get("embedding"):
                meta["embedding"] = old_ck.metadata["embedding"]
            return meta

        # 3. 插入所有新 chunks
        chunk_db_ids: list[int] = []

        def _maybe_split_chapter(ch: dict[str, Any], base_idx: int) -> list[dict[str, Any]]:
            """如果 chapter content 超过向量化限制，拆分为多个 sub-chapters."""
            content = ch["content"]
            max_chars = Config.CHUNK_MAX_CHARS
            if len(content) <= max_chars:
                return [{**ch, "_chunk_id": f"ch_{base_idx}"}]
            parts = _split_for_embedding(content, max_chars)
            return [
                {**ch, "content": part, "_chunk_id": f"ch_{base_idx}_part_{i}"}
                for i, part in enumerate(parts)
            ]

        def _insert_chunk(ch: dict[str, Any], ch_hash: str, old_ck: Chunk | None) -> int:
            merged = self._merge_image_descriptions(
                ch["content"], ch.get("title", ""), image_descriptions
            )
            chunk_id = ch.get("_chunk_id", f"ch_{len(chunk_db_ids)}")
            ck = Chunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                content=merged,
                chunk_type="text",
                chapter_title=ch["title"],
                metadata=_build_metadata(ch["title"], ch_hash, old_ck),
            )
            db_id = self.db.insert_chunk(ck)
            chunk_db_ids.append(db_id)
            return db_id

        for ch, old_cks in unchanged:
            base_idx = len(chunk_db_ids)
            subs = _maybe_split_chapter(ch, base_idx)
            # 只有当新旧 sub-chunk 数量一致时才按索引复用 embedding
            # （_split_for_embedding 是确定性函数，content_hash 不变则拆分结果不变）
            if old_cks and len(subs) == len(old_cks):
                for sub, old_ck in zip(subs, old_cks):
                    _insert_chunk(sub, sub["content_hash"], old_ck)
            else:
                for sub in subs:
                    _insert_chunk(sub, sub["content_hash"], None)
        for ch in changed:
            base_idx = len(chunk_db_ids)
            for sub in _maybe_split_chapter(ch, base_idx):
                _insert_chunk(sub, sub["content_hash"], None)
        for ch in added:
            base_idx = len(chunk_db_ids)
            for sub in _maybe_split_chapter(ch, base_idx):
                _insert_chunk(sub, sub["content_hash"], None)

        if not chapters:
            merged = self._merge_image_descriptions(full_text, "", image_descriptions)
            ck = Chunk(
                doc_id=doc_id,
                chunk_id="full",
                content=merged,
                chunk_type="text",
                chapter_title="",
                metadata={
                    "content_hash": "",
                    "extracted_entities": entity_dicts,
                    "extracted_relations": relation_dicts,
                    "image_descriptions": image_descriptions,
                },
            )
            db_id = self.db.insert_chunk(ck)
            chunk_db_ids.append(db_id)

        # 4. 更新 chunk 状态为 embedded（向量化由独立 stage6 完成）
        for db_id in chunk_db_ids:
            self.db.update_chunk_status(db_id, "embedded")

    def _merge_image_descriptions(
        self,
        content: str,
        chapter_title: str,
        image_descriptions: list[dict[str, Any]],
    ) -> str:
        """将图片文字化结果合并到 chunk 内容中.

        按 chapter_title 过滤，将 content 中的 Markdown 图片引用
        ``![img_id](path)`` 原位替换为 ``[img_id] description``；
        对于没有 Markdown 引用但有 description 的图片，在末尾追加兜底。
        """
        if not image_descriptions:
            return content

        chapter_images = [
            d for d in image_descriptions if d.get("chapter_title", "") == chapter_title
        ]
        if not chapter_images:
            return content

        # 1. 原位替换 Markdown 图片引用
        for d in chapter_images:
            img_id = d.get("image_id", "unknown")
            img_desc = d.get("description", "")
            content = re.sub(
                rf"!\[{re.escape(img_id)}\]\([^)]*\)",
                f"[{img_id}] {img_desc}",
                content,
            )

        # 2. 兜底：未匹配到 Markdown 引用的图片在末尾追加
        unmatched: list[dict[str, Any]] = []
        for d in chapter_images:
            img_id = d.get("image_id", "unknown")
            if not re.search(rf"\[{re.escape(img_id)}\] ", content):
                unmatched.append(d)

        if unmatched:
            parts = [content, "\n\n【图片内容】"]
            for d in unmatched:
                img_id = d.get("image_id", "unknown")
                img_desc = d.get("description", "")
                parts.append(f"[{img_id}] {img_desc}")
            content = "\n".join(parts)

        return content

    def close(self) -> None:
        """关闭资源."""
        if self.parser_client:
            self.parser_client = None
        if self.llm_batch:
            self.llm_batch.close()
