import pytest
from core.agentic_search import AgenticSearchPlanner, SearchStep
from core.llm_chat_client import BaseLLMClient


@pytest.fixture(autouse=True)
def _mock_llm_config(monkeypatch):
    """Provide a dummy LLM configuration so the default client can instantiate."""
    from core import config as config_module

    monkeypatch.setattr(config_module.Config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config_module.Config, "LLM_BASE_URL", "https://test.example.com/v1")
    monkeypatch.setattr(config_module.Config, "LLM_MODEL", "test-model")


def test_search_step_to_dict():
    step = SearchStep(
        step_type="semantic",
        query="SPI reset sequence",
        params={"top_k": 5},
        reason="Find relevant passages",
    )
    d = step.to_dict()
    assert d["step_type"] == "semantic"
    assert d["query"] == "SPI reset sequence"
    assert d["params"] == {"top_k": 5}
    assert d["reason"] == "Find relevant passages"


def test_planner_parses_valid_json(monkeypatch):
    def fake_chat(self, messages, **kwargs):
        return (
            '[{"step_type": "semantic", "query": "SPI reset", '
            '"params": {"top_k": 5}, "reason": "find docs"}]'
        )

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    planner = AgenticSearchPlanner()
    steps = planner.plan("What is the SPI reset sequence?")
    assert len(steps) == 1
    assert steps[0].step_type == "semantic"
    assert steps[0].query == "SPI reset"
    assert steps[0].params == {"top_k": 5}


def test_planner_parses_multiple_steps(monkeypatch):
    def fake_chat(self, messages, **kwargs):
        return (
            '[{"step_type": "semantic", "query": "SPI module"}, '
            '{"step_type": "graph", "query": "SPI::Module", '
            '"params": {"entity_type": "Module"}}, '
            '{"step_type": "synthesize", "query": "Summarize"}]'
        )

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    planner = AgenticSearchPlanner()
    steps = planner.plan("Compare SPI modules")
    assert len(steps) == 3
    assert steps[0].step_type == "semantic"
    assert steps[1].step_type == "graph"
    assert steps[2].step_type == "synthesize"


def test_planner_ignores_unknown_step_type(monkeypatch):
    def fake_chat(self, messages, **kwargs):
        return '[{"step_type": "semantic", "query": "x"}, {"step_type": "invalid", "query": "y"}]'

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    planner = AgenticSearchPlanner()
    steps = planner.plan("?")
    assert len(steps) == 1
    assert steps[0].step_type == "semantic"


def test_planner_handles_non_json(monkeypatch):
    monkeypatch.setattr(
        "core.llm_chat_client.BaseLLMClient.chat",
        lambda self, messages, **kwargs: "not json",
    )
    planner = AgenticSearchPlanner()
    steps = planner.plan("?")
    assert steps == []


def test_planner_handles_non_list_json(monkeypatch):
    monkeypatch.setattr(
        "core.llm_chat_client.BaseLLMClient.chat",
        lambda self, messages, **kwargs: "{}",
    )
    planner = AgenticSearchPlanner()
    steps = planner.plan("?")
    assert steps == []


def test_planner_uses_custom_client(monkeypatch):
    class FakeClient(BaseLLMClient):
        def chat(self, messages, temperature=0.3, **kwargs):
            return '[{"step_type": "chapter", "query": "reset"}]'

    planner = AgenticSearchPlanner(client=FakeClient())
    steps = planner.plan("?")
    assert len(steps) == 1
    assert steps[0].step_type == "chapter"


def test_planner_prompt_placeholder_collision(monkeypatch):
    captured: list[str] = []

    def fake_chat(self, messages, **kwargs):
        captured.append(messages[1]["content"])
        return '[{"step_type": "semantic", "query": "x"}]'

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    planner = AgenticSearchPlanner()
    planner.plan("question with $context literal")
    assert "$context" in captured[0]


def test_planner_returns_empty_on_llm_error(monkeypatch):
    def fake_chat(self, messages, **kwargs):
        raise RuntimeError("LLM timeout")

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    planner = AgenticSearchPlanner()
    steps = planner.plan("?")
    assert steps == []


def test_planner_ignores_non_dict_params(monkeypatch):
    def fake_chat(self, messages, **kwargs):
        return '[{"step_type": "semantic", "query": "x", "params": "bad"}]'

    monkeypatch.setattr("core.llm_chat_client.BaseLLMClient.chat", fake_chat)
    planner = AgenticSearchPlanner()
    steps = planner.plan("?")
    assert steps[0].params == {}
