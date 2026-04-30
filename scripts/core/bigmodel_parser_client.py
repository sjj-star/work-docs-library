"""BigModel (智谱) 文件解析客户端 - Expert 模式.

⚠️ 厂商专用：本客户端仅适用于 BigModel (智谱) 的 Expert 文件解析服务，
    使用专有端点 /files/parser/create 和 /files/parser/result，非 OpenAI-compatible API。
    如需使用其他厂商的 PDF 解析服务，需另行实现对应客户端。

封装 /files/parser/create + /files/parser/result 异步 API
输出：Markdown 文本 + 提取的图片文件列表.
"""

import io
import logging
import time
import zipfile
from pathlib import Path

import requests

from .config import Config

logger = logging.getLogger(__name__)


class BigModelParserClient:
    """BigModel Expert 文件解析客户端."""

    DEFAULT_TIMEOUT = Config.PARSER_TIMEOUT
    MAX_POLL_RETRIES = Config.PARSER_MAX_RETRIES
    POLL_INTERVAL = Config.PARSER_POLL_INTERVAL

    def __init__(self, api_key: str | None = None) -> None:
        """初始化 BigModelParserClient."""
        self.api_key = api_key or self._resolve_api_key()
        self.base_url = "https://open.bigmodel.cn/api/paas/v4"
        if not self.api_key:
            raise RuntimeError(
                "Parser API key not configured. Set WORKDOCS_PARSER_API_KEY in .env or config.json"
            )
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _resolve_api_key() -> str:
        """解析 Parser API Key."""
        # 优先级：环境变量(Kimi注入) > config.json > .env
        import os

        env_val = os.getenv("parser.api_key", "")
        if env_val:
            return env_val
        # config.json
        from .config import _CONFIG_JSON

        val = _CONFIG_JSON.get("parser", {}).get("api_key", "")
        if val:
            return val
        # .env
        return os.getenv("WORKDOCS_PARSER_API_KEY", "")

    def create_task(self, file_path: str | Path, tool_type: str = "expert") -> str:
        """创建文件解析任务.

        Args:
            file_path: PDF 文件路径
            tool_type: expert | prime | lite

        Returns:
            task_id

        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/files/parser/create",
                headers=self.headers,
                files={"file": (file_path.name, f)},
                data={"tool_type": tool_type, "file_type": "PDF"},
                timeout=self.DEFAULT_TIMEOUT,
            )

        resp.raise_for_status()
        data = resp.json()
        task_id = data.get("task_id")
        if not task_id:
            raise RuntimeError(f"Failed to create task: {data}")
        logger.info(f"BigModel task created | task_id={task_id} | file={file_path.name}")
        return task_id

    def poll_result(
        self,
        task_id: str,
        format_type: str = "download_link",
    ) -> dict:
        """轮询获取解析结果.

        Args:
            task_id: 任务 ID
            format_type: text | download_link

        Returns:
            解析结果 dict，包含 parsing_result_url 或 content

        """
        for i in range(self.MAX_POLL_RETRIES):
            resp = requests.get(
                f"{self.base_url}/files/parser/result/{task_id}/{format_type}",
                headers=self.headers,
                timeout=self.DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status") or data.get("data", {}).get("status", "unknown")
            logger.debug(f"BigModel poll {i + 1}/{self.MAX_POLL_RETRIES}: {status}")

            if status == "succeeded":
                logger.info(f"BigModel parsing succeeded | task_id={task_id}")
                return data
            elif status in ("failed", "error"):
                raise RuntimeError(f"BigModel parsing failed: {data}")

            time.sleep(self.POLL_INTERVAL)

        raise RuntimeError(f"BigModel polling timeout | task_id={task_id}")

    def download_result(self, url: str) -> bytes:
        """下载解析结果 ZIP."""
        resp = requests.get(url, timeout=self.DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.content

    def parse_pdf(
        self,
        file_path: str | Path,
        output_dir: str | Path | None = None,
        tool_type: str = "expert",
    ) -> tuple[str, list[Path]]:
        """一键解析 PDF：上传 → 轮询 → 下载 → 解压 → 返回 Markdown + 图片.

        Args:
            file_path: PDF 文件路径
            output_dir: 输出目录（默认：knowledge_base/parsed/<doc_id>/）
            tool_type: expert | prime | lite

        Returns:
            (markdown_text, image_paths)

        """
        file_path = Path(file_path)

        # 1. 创建任务
        task_id = self.create_task(file_path, tool_type=tool_type)

        # 2. 轮询结果
        result = self.poll_result(task_id, format_type="download_link")
        url = result.get("parsing_result_url") or result.get("data", {}).get("parsingResultUrl")
        if not url:
            raise RuntimeError("No download URL in response")

        # 3. 下载 ZIP
        zip_bytes = self.download_result(url)
        logger.info(f"BigModel result downloaded | size={len(zip_bytes)} bytes")

        # 4. 确定输出目录
        if output_dir is None:
            output_dir = Config.DB_PATH.parent / "parsed" / task_id
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 5. 解压 ZIP
        md_text = ""
        images = []
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for name in z.namelist():
                info = z.getinfo(name)
                if info.is_dir():
                    continue
                target = output_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "wb") as f:
                    f.write(z.read(name))

                if name.endswith(".md"):
                    md_text = target.read_text(encoding="utf-8")
                elif name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                    images.append(target)

        logger.info(
            f"BigModel parse complete | md_chars={len(md_text)} | images={len(images)} | "
            f"output={output_dir}"
        )
        return md_text, images
