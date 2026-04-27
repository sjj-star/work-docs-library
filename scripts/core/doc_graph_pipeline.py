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

from .batch_clients import BigModelBatchClient, KimiBatchClient
from .bigmodel_parser_client import BigModelParserClient
from .config import Config
from .db import KnowledgeDB
from .embedding_client import EmbeddingClient
from .graph_store import (
    ALL_NODE_TYPES,
    ALL_REL_TYPES,
    NODE_REGISTER,
    NODE_SIGNAL,
    GraphEntity,
    GraphRelation,
    GraphStore,
    NetworkXGraphStore,
)
from .models import Chunk, Document
from .vector_index import VectorIndex

logger = logging.getLogger(__name__)


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

        for i, line in enumerate(lines):
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
        - # 和第一个 ## 之间的文字归入第一个 ## 的 preface
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

        # 递归传播 preface：任何有子节点的节点，其 content 归入第一个子节点
        def _propagate_preface(node: ChapterNode) -> None:
            for child in node.children:
                _propagate_preface(child)
            if node.children and node.content:
                first_child = node.children[0]
                first_child.content = (node.content + "\n\n" + first_child.content).strip()
                node.content = ""

        for root in roots:
            _propagate_preface(root)

        return roots

    @classmethod
    def _is_heading(cls, line: str) -> tuple[int, str] | None:
        """判断一行是否为标题，返回 (level, title) 或 None."""
        # Markdown 标题: # ## ###
        if line.startswith("#"):
            level = 0
            for c in line:
                if c == "#":
                    level += 1
                else:
                    break
            title = line[level:].strip()
            if title:
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

        # 数字编号标题: 1. 概述, 1.1 特性, 2. 寄存器
        m = re.match(r"^(\d+(?:\.\d+)*)(?:\s+|[\.\)\:])\s*(.+)", line)
        if m:
            number = m.group(1)
            title = m.group(2).strip()
            level = number.count(".") + 1
            if title and len(title) < 100:
                # 拒绝日期格式（如 "16 March 2001" → number=16, title="March 2001"）
                if re.match(
                    r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$",
                    title,
                    re.I,
                ):
                    return None
                if re.match(r"^\d{4}-\d{2}-\d{2}$", title):
                    return None
                # 拒绝纯数字标题（页码噪声如 "1 1" → title="1"）
                if re.match(r"^\d+$", title):
                    return None
                return (level, f"{number} {title}")

        # 中文编号: 一、概述, （一）特性
        m = re.match(r"^[一二三四五六七八九十]+[、\.\s].{2,50}", line)
        if m:
            return (1, line.strip())

        return None


# ---------------------------------------------------------------------------
# Batch 构建器：按 Markdown 层级控制 batch 粒度
# ---------------------------------------------------------------------------


class BatchBuilder:
    """按 # / ## 层级构建 LLM batch 请求."""

    @classmethod
    def _collect_content(cls, node: ChapterNode) -> str:
        """递归收集节点自身 + 所有子孙的 content.

        将子节点的 heading 和 content 一并合并，保持层级结构可读性。
        """
        parts = [node.content] if node.content else []
        for child in node.children:
            child_content = cls._collect_content(child)
            if child_content:
                parts.append(f"{'#' * child.level} {child.title}\n\n{child_content}")
        return "\n\n".join(parts).strip()

    @classmethod
    def _find_chunk_nodes(cls, node: ChapterNode, chunk_level: int = 3) -> list[ChapterNode]:
        """从 node 开始，找到所有作为 chunk 单位的子节点.

        规则：
        - 如果 node 的 level >= chunk_level，node 本身是一个 chunk
        - 否则，递归到子节点中找
        - 如果 node 没有满足条件的子节点，node 本身降级作为 chunk
        """
        if node.level >= chunk_level:
            return [node]

        chunks: list[ChapterNode] = []
        for child in node.children:
            chunks.extend(cls._find_chunk_nodes(child, chunk_level))

        # 如果没有子节点满足条件，当前节点降级作为 chunk
        if not chunks:
            return [node]

        return chunks

    @classmethod
    def build_batches(
        cls,
        tree_chapters: list[ChapterNode],
        max_chars: int,
    ) -> list[list[dict[str, str]]]:
        """构建 batch 请求列表.

        规则：
        1. 每个 # 章节对应一个独立的 batch 请求列表（硬边界，不跨 # 合并）
        2. 在 # 内部，以 ### 为基本 chunk 单位（包含其所有子孙 content）
        3. 任何有子节点的节点，其 content 归入第一个子节点（preface 传播）
        4. 如果当前 ### < 50% max_chars，尝试与下一个同级 ### 合并
        5. 如果单个 ### 或合并后的块 > max_chars，按句子边界切分为 sub-batch
        6. 没有 ### 的文档：以 ## 为 chunk 单位；没有 ## 的文档：# 内容本身作为一个 batch

        Args:
            tree_chapters: ChapterParser.parse_tree() 的输出
            max_chars: 每批最大字符数

        Returns:
            每个内部列表代表一个 batch，包含一个或多个 chunk dict

        """
        batches: list[list[dict[str, str]]] = []
        half_max = max_chars // 2

        for root in tree_chapters:
            if not root.children:
                # 没有子节点，root 本身作为一个 batch
                content = cls._collect_content(root)
                chunks = cls._split_if_needed(root.title, content, max_chars)
                for chunk_content in chunks:
                    batches.append([{"title": root.title, "content": chunk_content}])
                continue

            # root 的直接子节点是 ##（章节），每个 ## 是一个硬边界
            for section in root.children:
                if not section.children:
                    # section 没有子节点，section 本身作为一个 batch
                    content = cls._collect_content(section)
                    chunks = cls._split_if_needed(section.title, content, max_chars)
                    for chunk_content in chunks:
                        batches.append([{"title": section.title, "content": chunk_content}])
                    continue

                # 在 section 内部，找到所有 level=3 作为 chunk 单位
                chunk_nodes = cls._find_chunk_nodes(section, chunk_level=3)

                i = 0
                while i < len(chunk_nodes):
                    node = chunk_nodes[i]
                    node_content = cls._collect_content(node)
                    current_chunks: list[dict[str, str]] = []
                    current_len = len(node_content)
                    current_chunks.append({"title": node.title, "content": node_content})

                    # 尝试合并下一个同级节点（如果当前 < 50% max_chars）
                    j = i + 1
                    while j < len(chunk_nodes):
                        next_node = chunk_nodes[j]
                        if next_node.level != node.level:
                            break
                        next_content = cls._collect_content(next_node)
                        if current_len + len(next_content) <= max_chars:
                            current_chunks.append(
                                {"title": next_node.title, "content": next_content}
                            )
                            current_len += len(next_content)
                            j += 1
                            # 如果已经 >= 50%，停止合并
                            if current_len >= half_max:
                                break
                        else:
                            break

                    # 检查总长度是否超过 max_chars
                    if current_len > max_chars:
                        # 需要切分（通常不会发生，因为合并时已检查）
                        combined_text = "\n\n".join(c["content"] for c in current_chunks)
                        splits = cls._split_by_sentences(combined_text, max_chars)
                        for idx, split in enumerate(splits):
                            batches.append(
                                [
                                    {
                                        "title": f"{node.title} (part {idx + 1}/{len(splits)})",
                                        "content": split,
                                    }
                                ]
                            )
                    else:
                        batches.append(current_chunks)

                    i = j

        return batches

    @classmethod
    def _split_if_needed(cls, title: str, content: str, max_len: int) -> list[str]:
        """如果内容超过 max_len，按句子切分."""
        if len(content) <= max_len:
            return [content]
        return cls._split_by_sentences(content, max_len)

    @classmethod
    def _split_by_sentences(cls, text: str, max_len: int) -> list[str]:
        """按句子边界切分文本."""
        sentences = re.split(r"(?<=[。！？\.\!\?])\s+|\n\n+", text)
        chunks: list[str] = []
        current = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(current) + len(s) + 2 > max_len:
                if current:
                    chunks.append(current)
                current = s
            else:
                current = (current + "\n\n" + s).strip() if current else s
        if current:
            chunks.append(current)
        return chunks if chunks else [text[:max_len]]


# ---------------------------------------------------------------------------
# 表格解析器：从 Markdown 表格提取结构化数据
# ---------------------------------------------------------------------------


class TableParser:
    """解析 Markdown 表格为结构化数据."""

    @classmethod
    def extract_tables(cls, text: str) -> list[dict[str, Any]]:
        """从文本中提取所有 Markdown 表格.

        Returns:
            [{"headers": [...], "rows": [[...], ...], "title": str, "context": str}, ...]

        """
        tables = []
        lines = text.splitlines()
        i = 0

        while i < len(lines):
            # 寻找表格开始（含 | 的行）
            if "|" in lines[i] and i + 1 < len(lines) and "---" in lines[i + 1]:
                # 找到表格标题（前面的非空行）
                title = ""
                for j in range(max(0, i - 3), i):
                    candidate = lines[j].strip()
                    if candidate and not candidate.startswith("|"):
                        title = candidate
                        break

                # 解析表头
                headers = [h.strip() for h in lines[i].split("|") if h.strip()]

                # 跳过分隔行
                i += 2

                # 解析数据行
                rows = []
                while i < len(lines) and "|" in lines[i]:
                    row = [c.strip() for c in lines[i].split("|") if c.strip() or c == ""]
                    # 处理空单元格
                    cells = lines[i].split("|")
                    row = []
                    for cell in cells:
                        stripped = cell.strip()
                        if stripped or cell:  # 保留空字符串表示空单元格
                            row.append(stripped)
                    # 去掉首尾空元素（由行首/行尾的 | 产生）
                    while row and row[0] == "":
                        row.pop(0)
                    while row and row[-1] == "":
                        row.pop()
                    if row:
                        rows.append(row)
                    i += 1

                # 上下文：表格前后的文本
                context_start = max(0, i - len(rows) - 5)
                context_end = min(len(lines), i + 3)
                context = "\n".join(lines[context_start:context_end]).strip()

                tables.append(
                    {
                        "headers": headers,
                        "rows": rows,
                        "title": title,
                        "context": context,
                    }
                )
            else:
                i += 1

        return tables


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

    def _build_batch_requests(self, batches, image_base_dir):
        requests = []
        for i, batch in enumerate(batches):
            chapter_text = "\n\n---\n\n".join(
                f"## {ch['title']}\n\n{ch['content']}" for ch in batch
            )
            user_prefix = self._user_template.replace("{{chapters}}", chapter_text).replace(
                "{{images}}", ""
            )

            # 流式解析 Markdown 图片引用，构建 multimodal content
            mm_content, image_meta = self._build_multimodal_content(
                chapter_text, image_base_dir, id_prefix=f"batch_{i}"
            )

            # 在 content 数组开头插入 user template 前缀
            content: list[dict[str, Any]] = [{"type": "text", "text": user_prefix}]
            content.extend(mm_content)

            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": content},
            ]
            requests.append(
                {
                    "custom_id": f"batch_{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": Config.LLM_MODEL,
                        "messages": messages,
                        "response_format": {"type": "json_object"},
                    },
                }
            )
            logger.info(f"Batch {i} 构建完成 | images={len(image_meta)}")
        return requests

    def extract_from_batches(self, batches, image_base_dir=None, doc_id=""):
        """extract_from_batches 函数."""
        if not self.batch_client or not batches:
            return [], [], []
        image_base_dir = image_base_dir or Path()
        requests = self._build_batch_requests(batches, image_base_dir)
        logger.info(f"提交 Batch 请求 | requests={len(requests)}")
        try:
            results = self.batch_client.submit_parallel_batches(requests)
        except Exception as e:
            logger.error(f"Batch 请求失败 | error={e}")
            return [], [], []

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
                cleaned = re.sub(r",(\s*[}\]])", r"", cleaned)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning(f"JSON 解析失败 | raw={raw[:200]}...")
                return None

    def extract_from_table(self, table, doc_id=""):
        """extract_from_table 函数."""
        entities, relations = [], []
        headers = [h.lower() for h in table.get("headers", [])]
        rows = table.get("rows", [])
        title = table.get("title", "")
        if not rows:
            return entities, relations

        is_register_table = any(
            k in " ".join(headers) for k in ["register", "address", "offset", "field", "bit"]
        )
        is_signal_table = any(
            k in " ".join(headers) for k in ["signal", "pin", "direction", "width"]
        )

        if is_register_table:
            name_idx = self._find_col(headers, ["register", "name", "寄存器"])
            addr_idx = self._find_col(headers, ["address", "offset", "addr", "地址"])
            desc_idx = self._find_col(headers, ["description", "desc", "描述"])
            for row in rows:
                if name_idx is None or name_idx >= len(row):
                    continue
                reg_name = row[name_idx].strip()
                if not reg_name:
                    continue
                props = {"description": ""}
                if addr_idx is not None and addr_idx < len(row):
                    props["address_offset"] = row[addr_idx].strip()
                if desc_idx is not None and desc_idx < len(row):
                    props["description"] = row[desc_idx].strip()
                entities.append(
                    GraphEntity(
                        entity_type=NODE_REGISTER,
                        name=reg_name,
                        properties=props,
                        source_doc_ids={doc_id},
                        source_chapter=title,
                    )
                )

        elif is_signal_table:
            name_idx = self._find_col(headers, ["signal", "name", "pin", "信号"])
            dir_idx = self._find_col(headers, ["direction", "type", "方向"])
            width_idx = self._find_col(headers, ["width", "size", "位宽"])
            desc_idx = self._find_col(headers, ["description", "desc", "描述"])
            for row in rows:
                if name_idx is None or name_idx >= len(row):
                    continue
                sig_name = row[name_idx].strip()
                if not sig_name:
                    continue
                props = {}
                if dir_idx is not None and dir_idx < len(row):
                    props["direction"] = row[dir_idx].strip()
                if width_idx is not None and width_idx < len(row):
                    props["width"] = row[width_idx].strip()
                if desc_idx is not None and desc_idx < len(row):
                    props["description"] = row[desc_idx].strip()
                entities.append(
                    GraphEntity(
                        entity_type=NODE_SIGNAL,
                        name=sig_name,
                        properties=props,
                        source_doc_ids={doc_id},
                        source_chapter=title,
                    )
                )

        return entities, relations

    @staticmethod
    def _find_col(headers, keywords):
        for i, h in enumerate(headers):
            for kw in keywords:
                if kw in h:
                    return i
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

        # 客户端
        try:
            self.parser_client = BigModelParserClient()
            logger.info("BigModelParserClient 已初始化")
        except RuntimeError as e:
            logger.warning(f"BigModelParserClient 初始化失败: {e}")
            self.parser_client = None

        try:
            self.kimi_batch = KimiBatchClient()
            logger.info("KimiBatchClient 已初始化")
        except RuntimeError as e:
            logger.warning(f"KimiBatchClient 初始化失败: {e}")
            self.kimi_batch = None

        try:
            self.bigmodel_batch = BigModelBatchClient()
            logger.info("BigModelBatchClient 已初始化")
        except RuntimeError as e:
            logger.warning(f"BigModelBatchClient 初始化失败: {e}")
            self.bigmodel_batch = None

        # 解析器（仅 PDF，Office 文件暂时不支持）
        self.parsers = {
            ".pdf": PDFParser(),
        }

        # 子组件
        self.chapter_parser = ChapterParser()
        self.table_parser = TableParser()
        self.entity_extractor = EntityExtractor(self.kimi_batch)
        self.embedding_batch_client = self.bigmodel_batch

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

    def _process_one(
        self,
        file_path: str,
        dry_run: bool = False,
        force: bool = False,
    ) -> str | None:
        """处理单个文档."""
        # 确保每个文档使用干净的局部图谱
        if hasattr(self.graph, "clear"):
            self.graph.clear()
        suffix = Path(file_path).suffix.lower()
        parser = self.parsers.get(suffix)
        if not parser:
            logger.warning(f"不支持的文件格式 | file={file_path}")
            return None

        # 计算文档 ID
        file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
        doc_id = file_hash[:16]

        # 检查是否需要重新处理
        if not force:
            existing = self.db.get_document_by_path(file_path)
            if existing and existing.file_hash == file_hash and existing.status == "done":
                logger.info(f"跳过未变更文档 | file={file_path} | doc_id={doc_id}")
                return existing.doc_id

        # dry-run：只返回文档信息
        if dry_run:
            logger.info(f"Dry-run | file={file_path}")
            return doc_id

        logger.info(f"开始处理文档 | file={file_path} | doc_id={doc_id}")

        # 获取 PDF 实际页数
        pdf_page_count = 0
        if suffix == ".pdf":
            try:
                with fitz.open(str(file_path)) as doc:
                    pdf_page_count = len(doc)
            except Exception as e:
                logger.warning(f"获取 PDF 页数失败 | error={e}")

        # --- Step 1: BigModel Expert 解析（主路径）---
        extracted_text = ""
        bigmodel_images: list[Path] = []
        parsed_output_dir = Config.DB_PATH.parent / "parsed" / doc_id

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

        # BigModel 失败时回退到本地解析
        if not extracted_text:
            extracted_text, local_images = parser.parse(file_path, output_dir=parsed_output_dir)
            bigmodel_images = local_images
            logger.info(
                f"本地解析回退 | length={len(extracted_text)} chars | images={len(local_images)}"
            )

        # --- Step 2: 章节解析（树形结构）---
        tree_chapters = self.chapter_parser.parse_tree(extracted_text)
        logger.info(f"章节解析完成 | roots={len(tree_chapters)}")

        # --- Step 3: 表格解析（规则提取，不依赖 LLM）---
        tables = self.table_parser.extract_tables(extracted_text)
        logger.info(f"表格解析完成 | tables={len(tables)}")

        # --- Step 4: 图片整理（仅统计数量，不记录坐标）---
        logger.info(f"图片整理完成 | images={len(bigmodel_images)}")

        # 扁平化章节列表（用于 chunk 存储和向量化）
        new_chapters_flat: list[dict[str, Any]] = []
        for root in tree_chapters:
            if root.children:
                for child in root.children:
                    new_chapters_flat.append({"title": child.title, "content": child.content})
            else:
                new_chapters_flat.append({"title": root.title, "content": root.content})

        # --- Step 5: Batch 构建与增量分析 ---
        # 读取旧 chunks（用于增量更新）
        old_doc = self.db.get_document_by_path(file_path)
        old_chunks: list[Chunk] = []
        if old_doc:
            old_chunks = self.db.query_by_doc(old_doc.doc_id)

        # 按 content_hash 分类章节
        old_chunk_map = {ck.chapter_title: ck for ck in old_chunks}
        unchanged_chapters: list[tuple[dict[str, Any], Chunk]] = []
        changed_chapters: list[dict[str, Any]] = []
        added_chapters: list[dict[str, Any]] = []
        removed_chunks: list[Chunk] = []

        for ch in new_chapters_flat:
            ch_hash = hashlib.md5(ch["content"].encode()).hexdigest()[:16]
            ch["content_hash"] = ch_hash
            if ch["title"] in old_chunk_map:
                old_ck = old_chunk_map[ch["title"]]
                old_hash = old_ck.metadata.get("content_hash", "")
                if old_hash == ch_hash and not force:
                    unchanged_chapters.append((ch, old_ck))
                else:
                    changed_chapters.append(ch)
            else:
                added_chapters.append(ch)

        for title, old_ck in old_chunk_map.items():
            if title not in {c["title"] for c in new_chapters_flat}:
                removed_chunks.append(old_ck)

        logger.info(
            f"章节增量分析 | 未变={len(unchanged_chapters)} | "
            f"变更={len(changed_chapters)} | 新增={len(added_chapters)} | "
            f"删除={len(removed_chunks)}"
        )

        # 为变更/新增章节构建 batch
        process_nodes: list[ChapterNode] = []
        for ch in changed_chapters + added_chapters:
            process_nodes.append(ChapterNode(level=1, title=ch["title"], content=ch["content"]))
        batches = (
            BatchBuilder.build_batches(process_nodes, Config.LLM_BATCH_MAX_CHARS)
            if process_nodes
            else []
        )
        logger.info(f"Batch 构建完成 | batches={len(batches)} | 增量模式")

        # --- Step 6: 实体提取（LLM Batch API）---
        all_entities: list[GraphEntity] = []
        all_relations: list[GraphRelation] = []
        all_image_descriptions: list[dict[str, Any]] = []

        # 6a: 复用未变章节的实体缓存
        for ch, old_ck in unchanged_chapters:
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

        # 6b: 从表格提取（规则解析）
        for table in tables:
            ents, rels = self.entity_extractor.extract_from_table(table, doc_id=doc_id)
            all_entities.extend(ents)
            all_relations.extend(rels)

        # 6c: 从变更/新增章节提取（LLM Batch API，multimodal：文本 + 图片流式处理）
        if self.kimi_batch and batches:
            ents, rels, img_descs = self.entity_extractor.extract_from_batches(
                batches=batches,
                image_base_dir=parsed_output_dir,
                doc_id=doc_id,
            )
            all_entities.extend(ents)
            all_relations.extend(rels)
            all_image_descriptions.extend(img_descs)

        logger.info(
            f"实体提取完成 | entities={len(all_entities)} | "
            f"relations={len(all_relations)} | image_descs={len(all_image_descriptions)}"
        )

        # --- Step 6: 构建图谱 ---
        # 去重：同名同类型的实体只保留一个（合并 properties）
        entity_map: dict[str, GraphEntity] = {}
        for e in all_entities:
            key = f"{e.entity_type}::{e.name}"
            if key in entity_map:
                # 合并 properties
                entity_map[key].properties.update(e.properties)
            else:
                entity_map[key] = e

        for e in entity_map.values():
            self.graph.add_entity(e)

        for r in all_relations:
            self.graph.add_relation(r)

        # --- Step 7: 保存 chunks 到 SQLite（增量模式，保留向量检索能力）---
        self._save_chunks_to_db(
            doc_id,
            file_path,
            file_hash,
            new_chapters_flat,
            extracted_text,
            unchanged_chapters,
            changed_chapters,
            added_chapters,
            removed_chunks,
            all_image_descriptions,
            all_entities,
            all_relations,
        )

        # --- Step 8: 持久化图谱 ---
        graph_path = Config.DB_PATH.parent / "graphs" / f"{doc_id}.json"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph.save(graph_path)
        logger.info(f"图谱已保存 | path={graph_path} | {self.graph.stats()}")

        # 如果 doc_id 变化（文件内容变了），删除旧图谱文件避免幽灵数据
        if old_doc and old_doc.doc_id != doc_id:
            old_graph_path = Config.DB_PATH.parent / "graphs" / f"{old_doc.doc_id}.json"
            if old_graph_path.exists():
                old_graph_path.unlink()
                logger.info(f"已删除旧图谱文件 | {old_graph_path}")

        # --- Step 9: 更新文档状态 ---
        self.db.upsert_document(
            Document(
                doc_id=doc_id,
                title=Path(file_path).stem,
                source_path=file_path,
                file_type=suffix.lstrip("."),
                total_pages=pdf_page_count,
                file_hash=file_hash,
                status="done",
            )
        )

        # 为没有 summary 的 chunk 生成默认 summary
        chunks = self.db.query_by_doc(doc_id)
        for ck in chunks:
            if not ck.summary:
                default_summary = ck.content[:200] + "..." if len(ck.content) > 200 else ck.content
                self.db.update_chunk_summary(ck.id, default_summary)

        logger.info(f"文档处理完成 | doc_id={doc_id} | mode=DOC_GRAPH_FLOW")
        return doc_id

    def _save_chunks_to_db(
        self,
        doc_id: str,
        file_path: str,
        file_hash: str,
        chapters: list[dict[str, Any]],
        full_text: str,
        unchanged: list[tuple[dict[str, Any], Chunk]] | None = None,
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
            meta: dict[str, Any] = {
                "content_hash": ch_hash,
                "extracted_entities": ch_ents,
                "extracted_relations": relation_dicts,
                "image_descriptions": image_descriptions,
            }
            if old_ck and old_ck.metadata.get("embedding"):
                meta["embedding"] = old_ck.metadata["embedding"]
            return meta

        # 3. 插入所有新 chunks
        chunk_db_ids: list[int] = []

        def _insert_chunk(ch: dict[str, Any], ch_hash: str, old_ck: Chunk | None) -> int:
            merged = self._merge_image_descriptions(
                ch["content"], ch.get("title", ""), image_descriptions
            )
            ck = Chunk(
                doc_id=doc_id,
                chunk_id=f"ch_{len(chunk_db_ids)}",
                content=merged,
                chunk_type="text",
                chapter_title=ch["title"],
                metadata=_build_metadata(ch["title"], ch_hash, old_ck),
            )
            db_id = self.db.insert_chunk(ck)
            chunk_db_ids.append(db_id)
            return db_id

        for ch, old_ck in unchanged:
            _insert_chunk(ch, ch["content_hash"], old_ck)
        for ch in changed:
            _insert_chunk(ch, ch["content_hash"], None)
        for ch in added:
            _insert_chunk(ch, ch["content_hash"], None)

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

        # 4. 向量化（复用未变章节的 embedding，重新向量化的变更/新增章节）
        try:
            db_chunks = self.db.query_by_doc(doc_id)
            reuse_pairs: list[tuple[int, list[float]]] = []
            reembed_pairs: list[tuple[str, int]] = []

            for ck in db_chunks:
                if not ck.content.strip():
                    continue
                emb = ck.metadata.get("embedding")
                if emb:
                    reuse_pairs.append((ck.id, emb))
                else:
                    reembed_pairs.append((ck.content, ck.id))

            if reuse_pairs:
                self.db.update_chunks_embedded_batch(reuse_pairs)
                self.vec.add_batch(reuse_pairs)

            if reembed_pairs:
                texts, valid_ids = zip(*reembed_pairs)
                if self.embedding_batch_client:
                    logger.info(f"开始批量向量化 | chunks={len(texts)} | client=BigModelBatch")
                    embeddings = self.embedding_batch_client.submit_embedding_batch(
                        texts=list(texts),
                        timeout=Config.LLM_BATCH_TIMEOUT,
                    )
                    items = list(zip(valid_ids, embeddings))
                    self.db.update_chunks_embedded_batch(items)
                    self.vec.add_batch(items)
                else:
                    embedder = EmbeddingClient()
                    for i in range(0, len(texts), Config.BATCH_SIZE):
                        batch_texts = list(texts[i : i + Config.BATCH_SIZE])
                        batch_ids = list(valid_ids[i : i + Config.BATCH_SIZE])
                        embeddings = embedder.embed(batch_texts)
                        items = list(zip(batch_ids, embeddings))
                        self.db.update_chunks_embedded_batch(items)
                        self.vec.add_batch(items)
                    embedder.close()
        except Exception as e:
            logger.warning(f"向量化失败 | doc_id={doc_id} | error={e}")

        # 5. 更新 chunk 状态为 done
        for db_id in chunk_db_ids:
            self.db.update_chunk_status(db_id, "done")

    def _merge_image_descriptions(
        self,
        content: str,
        chapter_title: str,
        image_descriptions: list[dict[str, Any]],
    ) -> str:
        """将图片文字化结果合并到 chunk 内容中."""
        if not image_descriptions:
            return content

        parts = [content, "\n\n【图片内容】"]
        for d in image_descriptions:
            img_id = d.get("image_id", "unknown")
            img_desc = d.get("description", "")
            parts.append(f"[{img_id}] {img_desc}")
        return "\n".join(parts)

    def close(self) -> None:
        """关闭资源."""
        if self.parser_client:
            self.parser_client = None
