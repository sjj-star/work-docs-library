"""APIClient 单元测试."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests
from core.api_client import (
    APIClient,
    APIError,
    AuthenticationError,
    BigModelProvider,
    KimiProvider,
    QuotaExceededError,
    RateLimitError,
    RetryPolicy,
)
from core.config import Config


class FakeResponse:
    """模拟 requests.Response."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {}
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def _make_client(
    provider: KimiProvider | BigModelProvider | None = None,
    max_attempts: int = 3,
    base_delay: float = 0.0,
    max_delay: float = 60.0,
    jitter: bool = False,
    respect_retry_after: bool = True,
) -> APIClient:
    provider = provider or KimiProvider(api_key="test-key", base_url="https://test.com")
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        jitter=jitter,
        respect_retry_after=respect_retry_after,
    )
    monkeypatch_policy = MagicMock(return_value=policy)
    provider.retry_policy = monkeypatch_policy  # type: ignore[method-assign]
    return APIClient(provider, timeout=10)


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch, tmp_path):
    """避免测试污染真实路径."""
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "workdocs.db")


def test_request_success(monkeypatch):
    """正常请求返回响应."""
    client = _make_client()

    class FakeSession:
        def request(self, method, url, **kwargs):
            return FakeResponse(json_data={"ok": True})

    monkeypatch.setattr(client, "_session", FakeSession())
    resp = client.get("/chat")
    assert resp.json()["ok"] is True


def test_request_retry_on_429_rate_limit(monkeypatch):
    """429 请求过多应重试."""
    client = _make_client(base_delay=0.0)
    call_count = [0]

    class FakeSession:
        def request(self, method, url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return FakeResponse(
                    status_code=429,
                    json_data={"error": {"message": "We're receiving too many requests"}},
                )
            return FakeResponse(json_data={"ok": True})

    monkeypatch.setattr(client, "_session", FakeSession())
    resp = client.get("/chat")
    assert resp.json()["ok"] is True
    assert call_count[0] == 2


def test_request_no_retry_on_quota_429(monkeypatch):
    """429 额度耗尽不应重试."""
    client = _make_client(base_delay=0.0)

    class FakeSession:
        def request(self, method, url, **kwargs):
            return FakeResponse(
                status_code=429,
                json_data={"error": {"message": "You've reached your usage limit for this period"}},
            )

    monkeypatch.setattr(client, "_session", FakeSession())
    with pytest.raises(QuotaExceededError):
        client.get("/chat")


def test_request_no_retry_on_401(monkeypatch):
    """401 不应重试."""
    client = _make_client(base_delay=0.0)

    class FakeSession:
        def request(self, method, url, **kwargs):
            return FakeResponse(
                status_code=401,
                json_data={"error": {"message": "Invalid API Key"}},
            )

    monkeypatch.setattr(client, "_session", FakeSession())
    with pytest.raises(AuthenticationError):
        client.get("/chat")


def test_request_retry_on_500(monkeypatch):
    """500 应重试."""
    client = _make_client(base_delay=0.0)
    call_count = [0]

    class FakeSession:
        def request(self, method, url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return FakeResponse(
                    status_code=500, json_data={"error": {"message": "server error"}}
                )
            return FakeResponse(json_data={"ok": True})

    monkeypatch.setattr(client, "_session", FakeSession())
    resp = client.get("/chat")
    assert resp.json()["ok"] is True
    assert call_count[0] == 2


def test_request_respects_retry_after(monkeypatch):
    """应优先使用 Retry-After 头."""
    client = _make_client(base_delay=1.0)
    sleep_calls = []

    class FakeSession:
        def request(self, method, url, **kwargs):
            return FakeResponse(
                status_code=429,
                json_data={"error": {"message": "too many requests"}},
                headers={"Retry-After": "2"},
            )

    monkeypatch.setattr(client, "_session", FakeSession())
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    with pytest.raises(RateLimitError):
        client.get("/chat")
    # 第一次重试前等待 2s，第二次重试前等待 2s
    assert sleep_calls[:2] == [pytest.approx(2.0, 0.01), pytest.approx(2.0, 0.01)]


def test_request_timeout_retry(monkeypatch):
    """超时异常应重试."""
    client = _make_client(base_delay=0.0)
    call_count = [0]

    class FakeSession:
        def request(self, method, url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise requests.Timeout("timeout")
            return FakeResponse(json_data={"ok": True})

    monkeypatch.setattr(client, "_session", FakeSession())
    resp = client.get("/chat")
    assert resp.json()["ok"] is True
    assert call_count[0] == 2


def test_compute_delay_with_jitter():
    """Jitter 应使延迟在 [base*2^attempt, max_delay] 范围内."""
    policy = RetryPolicy(
        max_attempts=3,
        base_delay=1.0,
        max_delay=10.0,
        jitter=True,
        respect_retry_after=False,
    )
    delay = policy.compute_delay(attempt=1)
    assert 2.0 <= delay <= 10.0


def test_compute_delay_capped_by_max():
    """延迟不超过 max_delay."""
    policy = RetryPolicy(
        max_attempts=3,
        base_delay=1.0,
        max_delay=5.0,
        jitter=False,
        respect_retry_after=False,
    )
    assert policy.compute_delay(attempt=10) == 5.0


def test_kimi_provider_classify_content_too_large():
    """KimiProvider 正确分类输入超长."""
    provider = KimiProvider(api_key="k", base_url="https://kimi.com")
    err = provider.classify(
        400,
        {"error": {"message": "Your request exceeded model token limit: 262144"}},
    )
    assert isinstance(err, APIError)
    assert err.status_code == 400


def test_bigmodel_provider_classify_rate_limit():
    """BigModelProvider 正确分类 429."""
    provider = BigModelProvider(api_key="k", base_url="https://bigmodel.com")
    err = provider.classify(429, {"error": {"message": "Rate limit exceeded"}})
    assert isinstance(err, RateLimitError)
