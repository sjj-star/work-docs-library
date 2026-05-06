"""Batch API 客户端 - 通用实现（服务商无感）.

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

    def __init__(self, api_key: str, base_url: str) -> None:
        """初始化 BaseBatchClient."""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._session = requests.Session()
        # 从 Config 读取运行时参数，保留实例属性供测试覆盖
        self.DEFAULT_POLL_INTERVAL = Config.BATCH_POLL_INTERVAL
        self.MAX_POLL_RETRIES = Config.BATCH_MAX_POLL_RETRIES
        self.MAX_FILE_SIZE_MB = Config.BATCH_MAX_FILE_SIZE_MB

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
        output_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        """提交 batch 请求并等待完成.

        Args:
            requests: JSONL 请求列表
            timeout: 超时时间（秒），默认从 Config.LLM_BATCH_TIMEOUT 读取
            poll_interval: 轮询间隔（秒）
            output_path: 结果文件保存路径（可选）

        Returns:
            结果列表，每个元素对应一个请求的结果

        """
        timeout = timeout or Config.LLM_BATCH_TIMEOUT
        poll_interval = poll_interval or self.DEFAULT_POLL_INTERVAL

        if not requests:
            return []

        # 1. 生成 JSONL 文件
        jsonl_path = (
            Path(Config.DB_PATH.parent) / Config.BATCH_TEMP_DIR / f"{uuid.uuid4().hex}.jsonl"
        )
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
                if output_path:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(output_text)
                    logger.info(f"Batch 结果已保存 | path={output_path}")
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
        max_file_size_mb: int | None = None,
        timeout: int | None = None,
        poll_interval: int | None = None,
        output_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        """将大量请求按 JSONL 大小拆分为多个并行 batch 提交。.

        当单个 JSONL 超过 max_file_size_mb 时，自动切分为多个子列表，
        每个子列表独立提交 batch 任务，最后合并结果。
        """
        if not requests:
            return []

        timeout = timeout or Config.LLM_BATCH_TIMEOUT
        poll_interval = poll_interval or self.DEFAULT_POLL_INTERVAL
        max_file_size_mb = (
            max_file_size_mb if max_file_size_mb is not None else self.MAX_FILE_SIZE_MB
        )
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
            return self.submit_and_wait(
                chunks[0], timeout=timeout, poll_interval=poll_interval, output_path=output_path
            )

        logger.info(f"并行 Batch 提交 | total_requests={len(requests)} | chunks={len(chunks)}")

        # 并行提交每个 chunk
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_results: list[dict[str, Any]] = []
        errors: list[str] = []
        max_workers = min(len(chunks), Config.BATCH_PARALLEL_WORKERS)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, chunk in enumerate(chunks):
                future = executor.submit(
                    self.submit_and_wait,
                    chunk,
                    timeout=timeout,
                    poll_interval=poll_interval,
                    output_path=output_path,
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
# 通用 Batch Client
# ---------------------------------------------------------------------------


class BatchClient(BaseBatchClient):
    """通用 Batch API 客户端（服务商无感）.

    通过配置参数适配不同服务商的 Batch API，无需硬编码服务商名称。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_endpoint: str | None = None,
        completion_window: str | None = None,
        files_url_path: str = "files",
        batches_url_path: str = "batches",
        download_url_template: str | None = None,
        auto_delete_input_file: bool = False,
        upload_mime_type: str | None = None,
    ) -> None:
        """初始化 BatchClient.

        Args:
            api_key: API 密钥，默认从 Config.LLM_API_KEY 读取
            base_url: API 基础 URL，默认从 Config.LLM_BASE_URL 读取
            batch_endpoint: Batch 任务 endpoint，默认从 Config.LLM_BATCH_ENDPOINT 读取
            completion_window: Batch 完成窗口，默认从 Config.LLM_BATCH_COMPLETION_WINDOW 读取
            files_url_path: 文件上传 URL 路径，默认 "files"
            batches_url_path: Batch 任务 URL 路径，默认 "batches"
            download_url_template: 文件下载 URL 模板，
                默认从 Config.BATCH_FILE_DOWNLOAD_TEMPLATE 读取
            auto_delete_input_file: 是否自动删除输入文件，默认 False
            upload_mime_type: 上传文件的 MIME 类型，None 表示不指定（由 requests 自动推断）
        """
        api_key = api_key or Config.LLM_API_KEY
        base_url = base_url or Config.LLM_BASE_URL
        if not api_key:
            raise RuntimeError("API key not configured")
        super().__init__(api_key, base_url)
        self.batch_endpoint = batch_endpoint or Config.LLM_BATCH_ENDPOINT
        self.completion_window = completion_window or Config.LLM_BATCH_COMPLETION_WINDOW
        self.files_url = f"{self.base_url}/{files_url_path.lstrip('/')}"
        self.batches_url = f"{self.base_url}/{batches_url_path.lstrip('/')}"
        self.download_url_template = download_url_template or Config.BATCH_FILE_DOWNLOAD_TEMPLATE
        self.auto_delete_input_file = auto_delete_input_file
        self.upload_mime_type = upload_mime_type

    def _upload_jsonl(self, file_path: Path) -> str:
        with open(file_path, "rb") as f:
            if self.upload_mime_type:
                files = {"file": (file_path.name, f, self.upload_mime_type)}
            else:
                files = {"file": (file_path.name, f)}
            resp = self._post(self.files_url, files=files)
        return resp["id"]

    def _create_batch(self, file_id: str) -> str:
        payload: dict[str, Any] = {
            "input_file_id": file_id,
            "endpoint": self.batch_endpoint,
        }
        if self.completion_window:
            payload["completion_window"] = self.completion_window
        if self.auto_delete_input_file:
            payload["auto_delete_input_file"] = True
        resp = self._post(self.batches_url, payload=payload, timeout=60)
        return resp["id"]

    def _get_batch_status(self, batch_id: str) -> dict[str, Any]:
        return self._get(f"{self.batches_url}/{batch_id}", timeout=60)

    def _download_file(self, file_id: str) -> str:
        url = self.download_url_template.format(base_url=self.base_url, file_id=file_id)
        resp = self._session.get(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
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
        endpoint = Config.EMBEDDING_BATCH_ENDPOINT
        requests = []
        for i, text in enumerate(texts):
            requests.append(
                {
                    "custom_id": f"embed_{i}",
                    "method": "POST",
                    "url": endpoint,
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
