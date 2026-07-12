"""统一 HTTP API 客户端基座.

为 Kimi / BigModel 两类服务商提供统一的请求入口、错误分类、重试退避与日志。
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import requests

from .config import Config

logger = logging.getLogger(__name__)


class APIError(Exception):
    """项目级 API 错误基类."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        """初始化 APIError."""
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body or {}


class AuthenticationError(APIError):
    """401/402/403 等认证或权限错误."""


class QuotaExceededError(APIError):
    """配额/额度耗尽，重试无效."""


class RateLimitError(APIError):
    """429 因请求频率过高，可重试."""


class ServerOverloadedError(APIError):
    """服务端过载，可重试."""


class RequestFormatError(APIError):
    """400 请求参数错误，不重试."""


class ContentTooLargeError(APIError):
    """400 输入内容超长，应由调用方拆分/截断后重试."""


class ContentFilterError(APIError):
    """400 内容安全拦截，不重试."""


class ServerError(APIError):
    """500/502/504 等服务端错误，可重试."""


class TransientError(APIError):
    """网络、超时、连接等临时错误，可重试."""


def _extract_message(body: dict[str, Any] | None) -> str:
    """从响应体中提取错误 message."""
    if not body:
        return ""
    error = body.get("error") if isinstance(body.get("error"), dict) else None
    if error:
        return error.get("message") or error.get("msg") or ""
    return body.get("message") or body.get("msg") or ""


def _match_keywords(text: str, keywords: tuple[str, ...]) -> bool:
    """不区分大小写匹配关键字."""
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


@dataclass(frozen=True)
class RetryPolicy:
    """重试策略."""

    max_attempts: int
    base_delay: float
    max_delay: float
    jitter: bool
    respect_retry_after: bool

    def is_retryable(self, error: APIError) -> bool:
        """判断某类错误是否可重试."""
        return isinstance(
            error,
            (RateLimitError, ServerOverloadedError, ServerError, TransientError),
        )

    def compute_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """计算第 attempt 次重试前的等待时间."""
        if retry_after is not None and retry_after > 0:
            delay = retry_after
        else:
            delay = self.base_delay * (2**attempt)
        if self.jitter:
            delay *= 1 + random.random()
        return min(delay, self.max_delay)


class APIProvider(ABC):
    """服务商抽象."""

    @property
    @abstractmethod
    def base_url(self) -> str:
        """API 基础 URL."""

    @abstractmethod
    def auth_headers(self) -> dict[str, str]:
        """返回认证头."""

    @abstractmethod
    def retry_policy(self) -> RetryPolicy:
        """返回默认重试策略."""

    @abstractmethod
    def classify(self, status_code: int, body: dict[str, Any] | None) -> APIError:
        """将 HTTP 响应分类为项目异常."""


class KimiProvider(APIProvider):
    """Kimi (Moonshot) Chat / Batch 服务商."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """初始化 KimiProvider."""
        self._api_key = api_key or Config.LLM_API_KEY
        self._base_url = (base_url or Config.LLM_BASE_URL).rstrip("/")

    @property
    def base_url(self) -> str:
        """返回 API 基础 URL."""
        return self._base_url

    def auth_headers(self) -> dict[str, str]:
        """返回认证头."""
        return {"Authorization": f"Bearer {self._api_key}"}

    def retry_policy(self) -> RetryPolicy:
        """返回默认重试策略."""
        return RetryPolicy(
            max_attempts=Config.HTTP_RETRY_MAX_ATTEMPTS,
            base_delay=Config.HTTP_RETRY_BASE_DELAY,
            max_delay=Config.HTTP_RETRY_MAX_DELAY,
            jitter=Config.HTTP_RETRY_JITTER,
            respect_retry_after=Config.HTTP_RETRY_RESPECT_RETRY_AFTER,
        )

    def classify(self, status_code: int, body: dict[str, Any] | None) -> APIError:
        """将 HTTP 响应分类为项目异常."""
        message = _extract_message(body)
        kwargs: dict[str, Any] = {"status_code": status_code, "body": body}

        if status_code == 401 or status_code == 402:
            return AuthenticationError(message, **kwargs)

        if status_code == 403:
            return QuotaExceededError(message, **kwargs)

        if status_code == 404:
            return RequestFormatError(message, **kwargs)

        if status_code == 429:
            quota_keywords = (
                "usage limit",
                "monthly usage limit",
                "reached your usage",
                "billing cycle",
            )
            if _match_keywords(message, quota_keywords):
                return QuotaExceededError(message, **kwargs)
            if _match_keywords(message, ("overloaded",)):
                return ServerOverloadedError(message, **kwargs)
            return RateLimitError(message, **kwargs)

        if status_code == 499:
            return TransientError(message or "Client closed request", **kwargs)

        if status_code in (500, 502, 503, 504):
            return ServerError(message, **kwargs)

        if status_code == 400:
            if _match_keywords(
                message,
                (
                    "exceeds limit",
                    "exceeded model token limit",
                    "message size",
                    "token limit",
                ),
            ):
                return ContentTooLargeError(message, **kwargs)
            if _match_keywords(message, ("high risk", "rejected", "content safety")):
                return ContentFilterError(message, **kwargs)
            return RequestFormatError(message, **kwargs)

        return APIError(message or f"HTTP {status_code}", **kwargs)


class BigModelProvider(APIProvider):
    """智谱 BigModel 服务商（Embedding / Parser）."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """初始化 BigModelProvider."""
        self._api_key = api_key or Config.EMBEDDING_API_KEY or Config.PARSER_API_KEY
        self._base_url = (base_url or Config.EMBEDDING_BASE_URL).rstrip("/")

    @property
    def base_url(self) -> str:
        """返回 API 基础 URL."""
        return self._base_url

    def auth_headers(self) -> dict[str, str]:
        """返回认证头."""
        return {"Authorization": f"Bearer {self._api_key}"}

    def retry_policy(self) -> RetryPolicy:
        """返回默认重试策略."""
        return RetryPolicy(
            max_attempts=Config.HTTP_RETRY_MAX_ATTEMPTS,
            base_delay=Config.HTTP_RETRY_BASE_DELAY,
            max_delay=Config.HTTP_RETRY_MAX_DELAY,
            jitter=Config.HTTP_RETRY_JITTER,
            respect_retry_after=Config.HTTP_RETRY_RESPECT_RETRY_AFTER,
        )

    def classify(self, status_code: int, body: dict[str, Any] | None) -> APIError:
        """将 HTTP 响应分类为项目异常."""
        message = _extract_message(body)
        kwargs: dict[str, Any] = {"status_code": status_code, "body": body}

        if status_code == 401:
            return AuthenticationError(message, **kwargs)

        if status_code == 403:
            return QuotaExceededError(message, **kwargs)

        if status_code == 404:
            return RequestFormatError(message, **kwargs)

        if status_code == 429:
            return RateLimitError(message, **kwargs)

        if status_code == 499:
            return TransientError(message or "Client closed request", **kwargs)

        if status_code in (500, 502, 504):
            return ServerError(message, **kwargs)

        if status_code == 503:
            return ServerOverloadedError(message, **kwargs)

        if status_code == 400:
            if _match_keywords(
                message,
                (
                    "too long",
                    "too large",
                    "exceed",
                    "max length",
                    "input length",
                    "content length",
                    "max tokens",
                ),
            ):
                return ContentTooLargeError(message, **kwargs)
            if _match_keywords(
                message,
                ("content filter", "safety", "inappropriate", "high risk"),
            ):
                return ContentFilterError(message, **kwargs)
            return RequestFormatError(message, **kwargs)

        return APIError(message or f"HTTP {status_code}", **kwargs)


class APIClient:
    """统一 HTTP 客户端."""

    def __init__(
        self,
        provider: APIProvider,
        *,
        timeout: float | None = None,
        user_agent: str | None = None,
    ) -> None:
        """初始化 APIClient."""
        self.provider = provider
        self.timeout = timeout or Config.HTTP_TIMEOUT
        self.user_agent = user_agent or "work-docs-library"
        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def close(self) -> None:
        """关闭底层 session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def _prepare_headers(self, extra: dict[str, str] | None) -> dict[str, str]:
        headers = {"User-Agent": self.user_agent}
        headers.update(self.provider.auth_headers())
        if extra:
            headers.update(extra)
        return headers

    def _parse_body(self, response: requests.Response) -> dict[str, Any] | None:
        try:
            return response.json()
        except Exception:
            return None

    def _retry_after(self, response: requests.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """执行 HTTP 请求，带统一重试与错误分类."""
        if path.startswith(("http://", "https://")):
            url = path
        else:
            url = f"{self.provider.base_url}{path}"
        prepared_headers = self._prepare_headers(headers)
        if "json" in kwargs and "Content-Type" not in prepared_headers:
            prepared_headers["Content-Type"] = "application/json"

        kwargs.setdefault("timeout", self.timeout)
        policy = self.provider.retry_policy()
        last_error: APIError | None = None

        for attempt in range(policy.max_attempts):
            try:
                session = self._get_session()
                response = session.request(
                    method,
                    url,
                    headers=prepared_headers,
                    **kwargs,
                )
                response.raise_for_status()
                return response
            except requests.HTTPError as exc:
                response = exc.response
                body = self._parse_body(response) if response is not None else None
                status = response.status_code if response is not None else None
                error = self.provider.classify(status or 0, body)
                last_error = error
                if not policy.is_retryable(error) or attempt == policy.max_attempts - 1:
                    raise error from exc
                retry_after = (
                    self._retry_after(response)
                    if policy.respect_retry_after and response is not None
                    else None
                )
                delay = policy.compute_delay(attempt, retry_after)
                logger.warning(
                    f"API request failed | provider={self.provider.__class__.__name__} "
                    f"| method={method} | path={path} | status={status} | "
                    f"message={error.message!r} | attempt={attempt + 1}/"
                    f"{policy.max_attempts} | retry_after={delay:.2f}s"
                )
                time.sleep(delay)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = TransientError(str(exc))
                if attempt == policy.max_attempts - 1:
                    raise last_error from exc
                delay = policy.compute_delay(attempt)
                logger.warning(
                    f"API transient error | provider={self.provider.__class__.__name__} "
                    f"| method={method} | path={path} | error={exc} | "
                    f"attempt={attempt + 1}/{policy.max_attempts} | retry_after={delay:.2f}s"
                )
                time.sleep(delay)

        # Defensive: should never reach here if max_attempts >= 1.
        raise last_error or APIError("Unknown API error")

    def get(
        self,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """发送 GET 请求."""
        return self.request("GET", path, headers=headers, **kwargs)

    def post(
        self,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """发送 POST 请求."""
        return self.request("POST", path, headers=headers, **kwargs)

    def delete(
        self,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """发送 DELETE 请求."""
        return self.request("DELETE", path, headers=headers, **kwargs)
