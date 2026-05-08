#!/usr/bin/env python3
"""同步 Embedding 向量化工具 — 用于 BigModel Batch API 卡住时的 fallback.

用法:
    PYTHONPATH=scripts venv/bin/python scripts/tools/sync_embed.py <doc_id>

说明:
    直接调用 BigModel 同步 Embedding API，绕过 Batch API。
    遵守 EMBED_ARRAY_MAX_SIZE 限制，使用数组模式请求以提升效率。
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.doc_graph_pipeline import DocGraphPipeline, split_text_by_paragraphs
from core.embedding_client import EmbeddingClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("sync_embed")


def main() -> None:
    """同步 Embedding 向量化入口."""
    parser = argparse.ArgumentParser(description="同步 Embedding 向量化")
    parser.add_argument("doc_id", help="文档 ID")
    args = parser.parse_args()

    pipeline = DocGraphPipeline()
    embedder = EmbeddingClient()

    try:
        db_chunks = pipeline.db.query_by_doc(args.doc_id)
        all_items: list[tuple[int, list[float]]] = []

        # 按 EMBED_ARRAY_MAX_SIZE 分批，减少 API 调用次数
        # BigModel embedding-3 单条请求最多 3072 tokens，保守限制 batch_size=1
        batch_size = 1
        texts_batch: list[str] = []
        ids_batch: list[int] = []

        failed_ids: list[int] = []
        for ck in db_chunks:
            if ck.metadata.get("embedding"):
                all_items.append((ck.id, ck.metadata["embedding"]))
                continue
            texts_batch.append(ck.content)
            ids_batch.append(ck.id)

            if len(texts_batch) >= batch_size:
                logger.info(f"提交同步 Embedding | batch={len(texts_batch)}")
                try:
                    embs = embedder.embed(texts_batch)
                    for db_id, emb in zip(ids_batch, embs):
                        all_items.append((db_id, emb))
                except Exception as e:
                    text = texts_batch[0]
                    logger.warning(
                        f"Embedding 失败 | db_id={ids_batch[0]} | "
                        f"chars={len(text)} | error={e}"
                    )
                    if len(text) > 6000:
                        # 超长文本：切分后分别嵌入，取平均
                        parts = split_text_by_paragraphs(text, 1500)
                        logger.info(f"切分超长文本 | db_id={ids_batch[0]} | parts={len(parts)}")
                        part_embs: list[list[float]] = []
                        for part in parts:
                            try:
                                pe = embedder.embed([part])
                                if pe:
                                    part_embs.append(pe[0])
                            except Exception as pe_err:
                                logger.warning(
                                    f"切分后嵌入仍失败 | chars={len(part)} | "
                                    f"error={pe_err}"
                                )
                        if part_embs:
                            avg_emb = [
                                sum(x[i] for x in part_embs) / len(part_embs)
                                for i in range(len(part_embs[0]))
                            ]
                            all_items.append((ids_batch[0], avg_emb))
                        else:
                            failed_ids.extend(ids_batch)
                    else:
                        failed_ids.extend(ids_batch)
                texts_batch = []
                ids_batch = []

        if texts_batch:
            logger.info(f"提交同步 Embedding | batch={len(texts_batch)} (final)")
            try:
                embs = embedder.embed(texts_batch)
                for db_id, emb in zip(ids_batch, embs):
                    all_items.append((db_id, emb))
            except Exception as e:
                text = texts_batch[0]
                logger.warning(
                    f"Embedding 失败 | db_id={ids_batch[0]} | "
                    f"chars={len(text)} | error={e}"
                )
                if len(text) > 6000:
                    parts = split_text_by_paragraphs(text, 1500)
                    logger.info(f"切分超长文本 | db_id={ids_batch[0]} | parts={len(parts)}")
                    part_embs: list[list[float]] = []
                    for part in parts:
                        try:
                            pe = embedder.embed([part])
                            if pe:
                                part_embs.append(pe[0])
                        except Exception as pe_err:
                            logger.warning(
                                f"切分后嵌入仍失败 | chars={len(part)} | "
                                f"error={pe_err}"
                            )
                    if part_embs:
                        avg_emb = [
                            sum(x[i] for x in part_embs) / len(part_embs)
                            for i in range(len(part_embs[0]))
                        ]
                        all_items.append((ids_batch[0], avg_emb))
                    else:
                        failed_ids.extend(ids_batch)
                else:
                    failed_ids.extend(ids_batch)

        if all_items:
            pipeline.db.update_chunks_embedded_batch(all_items)
            pipeline.vec.add_batch(all_items)

        for ck in db_chunks:
            pipeline.db.update_chunk_status(ck.id, "done")

        logger.info(f"同步 Embedding 完成 | doc_id={args.doc_id} | items={len(all_items)}")
        print(f"done | doc_id={args.doc_id} | embedded={len(all_items)}")
    finally:
        embedder.close()
        pipeline.close()


if __name__ == "__main__":
    main()
