"""Agentic search mechanism layer.

Provides query decomposition into SearchSteps. Execution of the steps is
intentionally left to the calling Agent / Skill so that strategies live in
Skills and mechanisms live in code.
"""

import json
import logging
from dataclasses import dataclass, field
from string import Template
from typing import Any

from .config import Config
from .llm_chat_client import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass
class SearchStep:
    """A single search step planned by AgenticSearchPlanner."""

    step_type: (
        str  # "semantic" | "hybrid" | "reranked" | "graph" | "chapter" | "metadata" | "synthesize"
    )
    query: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the step."""
        return {
            "step_type": self.step_type,
            "query": self.query,
            "params": self.params,
            "reason": self.reason,
        }


class AgenticSearchPlanner:
    """把复杂问题分解为 SearchStep 列表，但不执行它们."""

    def __init__(self, client: BaseLLMClient | None = None) -> None:
        """Initialize the planner with an optional chat client."""
        self.client = client or BaseLLMClient()

    @staticmethod
    def _load_prompt_template(name: str) -> Template:
        """Load a prompt file and return it as a string.Template for safe substitution."""
        path = Config.PROMPT_DIR / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return Template(path.read_text(encoding="utf-8"))

    @staticmethod
    def _parse_steps(raw: str) -> list[SearchStep]:
        """Parse LLM response into SearchStep objects."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Agentic planner returned non-JSON: {raw[:200]}")
            parsed = []

        if not isinstance(parsed, list):
            logger.warning(f"Agentic planner returned non-list: {type(parsed)}")
            parsed = []

        steps: list[SearchStep] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            step_type = item.get("step_type", "")
            if step_type not in {
                "semantic",
                "hybrid",
                "reranked",
                "graph",
                "chapter",
                "metadata",
                "synthesize",
            }:
                logger.warning(f"Agentic planner returned unknown step_type: {step_type}")
                continue
            steps.append(
                SearchStep(
                    step_type=step_type,
                    query=str(item.get("query", "")),
                    params=item.get("params", {}) if isinstance(item.get("params"), dict) else {},
                    reason=str(item.get("reason", "")),
                )
            )
        return steps

    def plan(self, question: str, context: dict[str, Any] | None = None) -> list[SearchStep]:
        """Decompose a complex question into SearchSteps."""
        system_template = self._load_prompt_template("agentic_search_system")
        user_template = self._load_prompt_template("agentic_search_user")
        user = user_template.safe_substitute(
            question=question,
            context=json.dumps(context or {}, ensure_ascii=False),
        )
        system = system_template.template  # system prompt has no placeholders
        try:
            raw = self.client.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except Exception:
            logger.exception("Agentic search planning failed")
            return []
        return self._parse_steps(raw)
