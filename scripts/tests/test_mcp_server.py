"""Tests for the MCP stdio server."""

import json
import os
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path

import mcp_server as mcp
import pytest


@pytest.fixture
def captured(monkeypatch):
    """Capture JSON-RPC responses sent by the server."""
    responses = []

    def _capture(response: dict) -> None:
        responses.append(response)

    monkeypatch.setattr(mcp, "_send_response", _capture)
    return responses


class TestJsonRpcProtocol:
    def test_initialize(self, captured):
        mcp._process_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert len(captured) == 1
        resp = captured[0]
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert "tools" in resp["result"]["capabilities"]

    def test_initialized_notification_is_silent(self, captured):
        mcp._process_request({"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert len(captured) == 0

    def test_tools_list_returns_only_whitelisted_tools(self, captured):
        mcp._process_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        assert len(captured) == 1
        tools = captured[0]["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == set(mcp.MCP_TOOL_MAP.keys())
        assert len(tools) == 11
        for tool in tools:
            assert "description" in tool
            assert tool["inputSchema"]["type"] == "object"

    def test_unknown_method(self, captured):
        mcp._process_request({"jsonrpc": "2.0", "id": 3, "method": "foo/bar", "params": {}})
        assert len(captured) == 1
        assert captured[0]["error"]["code"] == -32601

    def test_unknown_method_notification_is_silent(self, captured):
        mcp._process_request({"jsonrpc": "2.0", "method": "foo/bar"})
        assert len(captured) == 0


class TestToolCalls:
    def test_tool_call_dispatches_arguments(self, captured, monkeypatch):
        calls = []

        def _fake_tool(arguments: dict) -> dict:
            calls.append(arguments)
            return {"success": True, "echo": arguments}

        monkeypatch.setitem(mcp.MCP_TOOL_MAP, "ingest", _fake_tool)

        mcp._process_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "ingest",
                    "arguments": {"path": "/tmp/test.pdf", "dry_run": True},
                },
            }
        )
        assert len(captured) == 1
        assert len(calls) == 1
        assert calls[0]["path"] == "/tmp/test.pdf"
        assert calls[0]["dry_run"] is True
        result = json.loads(captured[0]["result"]["content"][0]["text"])
        assert result["success"] is True
        assert captured[0]["result"]["isError"] is False

    def test_tool_call_unknown_tool(self, captured):
        mcp._process_request(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "not_a_tool", "arguments": {}},
            }
        )
        assert len(captured) == 1
        assert captured[0]["result"]["isError"] is True
        result = json.loads(captured[0]["result"]["content"][0]["text"])
        assert result["success"] is False

    def test_tool_call_exception_becomes_error(self, captured, monkeypatch):
        def _boom(_arguments: dict) -> dict:
            raise RuntimeError("boom")

        monkeypatch.setitem(mcp.MCP_TOOL_MAP, "config", _boom)
        mcp._process_request(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "config", "arguments": {}},
            }
        )
        assert len(captured) == 1
        assert captured[0]["result"]["isError"] is True
        result = json.loads(captured[0]["result"]["content"][0]["text"])
        assert result["success"] is False
        assert "boom" in result["error"]


class TestStdioLoop:
    def test_main_reads_lines_until_eof(self, monkeypatch):
        lines = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
        ]
        monkeypatch.setattr(sys, "stdin", StringIO("\n".join(lines) + "\n"))

        stdout = StringIO()
        monkeypatch.setattr(sys, "stdout", stdout)

        mcp.main()

        responses = [json.loads(line) for line in stdout.getvalue().strip().split("\n")]
        assert len(responses) == 2
        assert responses[0]["id"] == 1
        assert responses[1]["id"] == 2
        assert len(responses[1]["result"]["tools"]) == 11


class TestIntegration:
    def test_subprocess_initialize_list_config(self, tmp_path, monkeypatch):
        """Start the real MCP server and exercise initialize/list/call config."""
        env = dict(os.environ) if "os" in globals() else {}
        # Avoid inheriting any .env keys in the subprocess for a deterministic test.
        for key in list(env.keys()):
            if key.startswith("WORKDOCS_"):
                env.pop(key, None)

        proc = subprocess.Popen(
            [sys.executable, "scripts/mcp_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
            env=env,
        )
        try:
            _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            init = _recv(proc, timeout=5)
            assert init["result"]["protocolVersion"] == "2024-11-05"

            _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

            _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            listed = _recv(proc, timeout=5)
            names = {t["name"] for t in listed["result"]["tools"]}
            assert names == set(mcp.MCP_TOOL_MAP.keys())

            _send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "config", "arguments": {}},
                },
            )
            called = _recv(proc, timeout=5)
            result = json.loads(called["result"]["content"][0]["text"])
            assert result["success"] is True
            assert "config_groups" in result
        finally:
            assert proc.stdin is not None
            proc.stdin.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def _send(proc, msg: dict) -> None:
    proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
    proc.stdin.flush()


def _recv(proc, timeout: float = 5) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if line:
            return json.loads(line)
        time.sleep(0.01)
    raise TimeoutError("No response from MCP server")
