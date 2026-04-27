"""Batch API 客户端 - Kimi + BigModel.

封装异步 batch 处理流程：JSONL 构造 → 上传 → 创建 batch → 轮询 → 下载结果.
"""

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests

from .config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _build_jsonl(requests: list[dict[str, Any]], output_path: Path) -> Path:
    """将请求列表写入 JSONL 文件."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    logger.info(f"JSONL 文件已生成 | path={output_path} | requests={len(requests)}")
    return output_path


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    """解析 JSONL 文本为列表."""
    results = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning(f"JSONL 解析失败 | line={line[:200]} | error={e}")
    return results


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseBatchClient(ABC):
    """Batch API 客户端抽象基类."""

    DEFAULT_POLL_INTERVAL = 10  # 轮询间隔（秒）
    MAX_POLL_RETRIES = 360  # 最大轮询次数（默认 3600 秒）
    MAX_FILE_SIZE_MB = 100  # 单个 JSONL 文件最大 100MB

    def __init__(self, api_key: str, base_url: str) -> None:
        """初始化 BaseBatchClient."""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._session = requests.Session()

    def _post(
        self, url: str, payload: dict | None = None, files: dict | None = None, timeout: int = 120
    ) -> dict[str, Any]:
        """发送 POST 请求."""
        if files:
            # 文件上传时不设置 Content-Type（requests 会自动设置 multipart）
            headers = {k: v for k, v in self.headers.items() if k.lower() != "content-type"}
            resp = self._session.post(url, headers=headers, files=files, timeout=timeout)
        else:
            resp = self._session.post(url, headers=self.headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _get(self, url: str, timeout: int = 120) -> dict[str, Any]:
        """发送 GET 请求."""
        resp = self._session.get(url, headers=self.headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, url: str, timeout: int = 60) -> dict[str, Any]:
        """发送 DELETE 请求."""
        resp = self._session.delete(url, headers=self.headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    # -----------------------------------------------------------------------
    # 子类必须实现
    # -----------------------------------------------------------------------

    @abstractmethod
    def _upload_jsonl(self, file_path: Path) -> str:
        """上传 JSONL 文件，返回 file_id."""
        ...

    @abstractmethod
    def _create_batch(self, file_id: str) -> str:
        """创建 batch 任务，返回 batch_id."""
        ...

    @abstractmethod
    def _get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """获取 batch 状态."""
        ...

    @abstractmethod
    def _download_file(self, file_id: str) -> str:
        """下载文件内容，返回文本."""
        ...

    @abstractmethod
    def _delete_file(self, file_id: str) -> None:
        """删除文件."""
        ...

    # -----------------------------------------------------------------------
    # 通用流程
    # -----------------------------------------------------------------------

    def submit_and_wait(
        self,
        requests: list[dict[str, Any]],
        timeout: int | None = None,
        poll_interval: int | None = None,
    ) -> list[dict[str, Any]]:
        """提交 batch 请求并等待完成.

        Args:
            requests: JSONL 请求列表
            timeout: 超时时间（秒），默认从 Config.LLM_BATCH_TIMEOUT 读取
            poll_interval: 轮询间隔（秒）

        Returns:
            结果列表，每个元素对应一个请求的结果

        """
        timeout = timeout or Config.LLM_BATCH_TIMEOUT
        poll_interval = poll_interval or self.DEFAULT_POLL_INTERVAL

        if not requests:
            return []

        # 1. 生成 JSONL 文件
        jsonl_path = Path(Config.DB_PATH.parent) / "batch_temp" / f"{uuid.uuid4().hex}.jsonl"
        _build_jsonl(requests, jsonl_path)

        # 检查文件大小
        file_size_mb = jsonl_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise RuntimeError(
                f"JSONL 文件过大 | size={file_size_mb:.1f}MB | max={self.MAX_FILE_SIZE_MB}MB"
            )

        file_id = None
        batch_id = None
        try:
            # 2. 上传文件
            file_id = self._upload_jsonl(jsonl_path)
            logger.info(f"Batch 文件已上传 | file_id={file_id}")

            # 3. 创建 batch
            batch_id = self._create_batch(file_id)
            logger.info(f"Batch 任务已创建 | batch_id={batch_id}")

            # 4. 轮询等待
            start_time = time.time()
            for attempt in range(self.MAX_POLL_RETRIES):
                status_info = self._get_batch_status(batch_id)
                status = status_info.get("status", "unknown")
                counts = status_info.get("request_counts", {})
                completed = counts.get("completed", 0)
                total = counts.get("total", len(requests))

                logger.info(
                    f"Batch 轮询 | batch_id={batch_id} | status={status} | "
                    f"completed={completed}/{total} | attempt={attempt + 1}"
                )

                if status == "completed":
                    break
                elif status in ("failed", "expired", "cancelled"):
                    raise RuntimeError(
                        f"Batch 任务异常终止 | status={status} | batch_id={batch_id}"
                    )

                if time.time() - start_time > timeout:
                    raise RuntimeError(
                        f"Batch 任务轮询超时 | batch_id={batch_id} | timeout={timeout}s"
                    )

                time.sleep(poll_interval)
            else:
                raise RuntimeError(f"Batch 任务轮询次数耗尽 | batch_id={batch_id}")

            # 5. 下载结果
            output_file_id = status_info.get("output_file_id")
            error_file_id = status_info.get("error_file_id")

            results = []
            if output_file_id:
                output_text = self._download_file(output_file_id)
                results = _parse_jsonl(output_text)
                logger.info(f"Batch 结果下载完成 | results={len(results)}")

            if error_file_id:
                error_text = self._download_file(error_file_id)
                errors = _parse_jsonl(error_text)
                if errors:
                    logger.warning(f"Batch 中存在错误请求 | errors={len(errors)}")

            return results

        finally:
            # 清理临时文件
            if jsonl_path.exists():
                jsonl_path.unlink()
                logger.debug(f"临时 JSONL 文件已删除 | path={jsonl_path}")
            # 清理服务器文件（可选）
            try:
                if file_id:
                    self._delete_file(file_id)
            except Exception as e:
                logger.warning(f"删除 batch 输入文件失败 | file_id={file_id} | error={e}")

    def submit_parallel_batches(
        self,
        requests: list[dict[str, Any]],
        max_file_size_mb: int = 100,
        timeout: int | None = None,
        poll_interval: int | None = None,
    ) -> list[dict[str, Any]]:
        """将大量请求按 JSONL 大小拆分为多个并行 batch 提交。.

        当单个 JSONL 超过 max_file_size_mb 时，自动切分为多个子列表，
        每个子列表独立提交 batch 任务，最后合并结果。
        """
        if not requests:
            return []

        timeout = timeout or Config.LLM_BATCH_TIMEOUT
        poll_interval = poll_interval or self.DEFAULT_POLL_INTERVAL
        max_bytes = max_file_size_mb * 1024 * 1024

        # 按预估 JSONL 大小切分 requests
        chunks: list[list[dict[str, Any]]] = []
        current_chunk: list[dict[str, Any]] = []
        current_bytes = 0

        for req in requests:
            req_bytes = (
                len(json.dumps(req, ensure_ascii=False).encode("utf-8")) + 1
            )  # +1 for newline
            if current_bytes + req_bytes > max_bytes and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [req]
                current_bytes = req_bytes
            else:
                current_chunk.append(req)
                current_bytes += req_bytes

        if current_chunk:
            chunks.append(current_chunk)

        if len(chunks) == 1:
            # 只有一个 chunk，直接走单 batch
            return self.submit_and_wait(chunks[0], timeout=timeout, poll_interval=poll_interval)

        logger.info(f"并行 Batch 提交 | total_requests={len(requests)} | chunks={len(chunks)}")

        # 并行提交每个 chunk
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_results: list[dict[str, Any]] = []
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
            future_to_idx = {}
            for idx, chunk in enumerate(chunks):
                future = executor.submit(
                    self.submit_and_wait,
                    chunk,
                    timeout=timeout,
                    poll_interval=poll_interval,
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info(f"Chunk {idx} 完成 | results={len(results)}")
                except Exception as e:
                    errors.append(f"Chunk {idx} 失败: {e}")
                    logger.error(f"Chunk {idx} 失败 | error={e}")

        if errors:
            raise RuntimeError(f"并行 Batch 部分失败 | errors={errors}")

        logger.info(f"并行 Batch 全部完成 | total_results={len(all_results)}")
        return all_results

    def close(self) -> None:
        """关闭会话."""
        self._session.close()


# ---------------------------------------------------------------------------
# Kimi Batch Client
# ---------------------------------------------------------------------------


class KimiBatchClient(BaseBatchClient):
    """Kimi Batch API 客户端.

    支持的模型: kimi-k2.5, kimi-k2.6
    限制:
    - temperature, top_p, n, presence_penalty, frequency_penalty 不可修改
    - JSONL 文件大小不超过 100MB
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """初始化 KimiBatchClient."""
        api_key = api_key or Config.LLM_API_KEY
        base_url = base_url or Config.LLM_BASE_URL
        if not api_key:
            raise RuntimeError("Kimi API key not configured")
        super().__init__(api_key, base_url)
        self.files_url = f"{self.base_url}/files"
        self.batches_url = f"{self.base_url}/batches"

    def _upload_jsonl(self, file_path: Path) -> str:
        with open(file_path, "rb") as f:
            resp = self._post(
                self.files_url,
                files={"file": (file_path.name, f)},
                timeout=120,
            )
        return resp["id"]

    def _create_batch(self, file_id: str) -> str:
        payload = {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }
        resp = self._post(self.batches_url, payload=payload, timeout=60)
        return resp["id"]

    def _get_batch_status(self, batch_id: str) -> dict[str, Any]:
        return self._get(f"{self.batches_url}/{batch_id}", timeout=60)

    def _download_file(self, file_id: str) -> str:
        resp = self._session.get(
            f"{self.files_url}/{file_id}/content",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.text

    def _delete_file(self, file_id: str) -> None:
        self._delete(f"{self.files_url}/{file_id}")


# ---------------------------------------------------------------------------
# BigModel Batch Client
# ---------------------------------------------------------------------------


class BigModelBatchClient(BaseBatchClient):
    """BigModel (智谱) Batch API 客户端.

    支持 LLM 和 Embedding 两种 endpoint:
    - LLM: /v4/chat/completions (GLM-4 系列)
    - Embedding: /v4/embeddings (Embedding-2, Embedding-3)

    价格: 标准 API 的 50%
    限制:
    - JSONL 文件大小不超过 100MB
    - 单个文件最多 50,000 个请求（Embedding 最多 10,000）
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """初始化 BigModelBatchClient."""
        api_key = api_key or Config.EMBEDDING_API_KEY
        base_url = base_url or Config.EMBEDDING_BASE_URL
        if not api_key:
            raise RuntimeError("BigModel API key not configured")
        super().__init__(api_key, base_url)
        self.files_url = f"{self.base_url}/files"
        self.batches_url = f"{self.base_url}/batches"

    def _upload_jsonl(self, file_path: Path) -> str:
        with open(file_path, "rb") as f:
            resp = self._post(
                self.files_url,
                files={"file": (file_path.name, f, "application/json")},
                timeout=120,
            )
        return resp["id"]

    def _create_batch(self, file_id: str, endpoint: str = "/v4/chat/completions") -> str:
        payload = {
            "input_file_id": file_id,
            "endpoint": endpoint,
            "auto_delete_input_file": True,
        }
        resp = self._post(self.batches_url, payload=payload, timeout=60)
        return resp["id"]

    def _get_batch_status(self, batch_id: str) -> dict[str, Any]:
        return self._get(f"{self.batches_url}/{batch_id}", timeout=60)

    def _download_file(self, file_id: str) -> str:
        resp = self._session.get(
            f"{self.files_url}/{file_id}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.text

    def _delete_file(self, file_id: str) -> None:
        self._delete(f"{self.files_url}/{file_id}")

    def submit_embedding_batch(
        self,
        texts: list[str],
        model: str | None = None,
        timeout: int | None = None,
    ) -> list[list[float]]:
        """提交 Embedding batch 请求.

        Args:
            texts: 需要向量化的文本列表
            model: 模型名称，默认使用 Config.EMBEDDING_MODEL
            timeout: 超时时间

        Returns:
            embedding 列表，与 texts 一一对应

        """
        model = model or Config.EMBEDDING_MODEL
        requests = []
        for i, text in enumerate(texts):
            requests.append(
                {
                    "custom_id": f"embed_{i}",
                    "method": "POST",
                    "url": "/v4/embeddings",
                    "body": {
                        "model": model,
                        "input": text,
                    },
                }
            )

        results = self.submit_and_wait(requests, timeout=timeout)

        # 解析 embedding 结果
        embeddings = []
        for r in results:
            custom_id = r.get("custom_id", "")
            idx = int(custom_id.split("_")[1]) if "_" in custom_id else len(embeddings)
            body = r.get("response", {}).get("body", {})
            data = body.get("data", [])
            if data:
                embeddings.append((idx, data[0].get("embedding", [])))
            else:
                embeddings.append((idx, []))

        # 按索引排序
        embeddings.sort(key=lambda x: x[0])
        return [emb for _, emb in embeddings]
