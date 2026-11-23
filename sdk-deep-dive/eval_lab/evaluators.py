"""Evaluators and target function shared by challenges 4, 5 and the gate.

One implementation, so the baseline, the candidate and the gate are scored by
exactly the same code.
"""
import pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parents[1]

import json
import re
import sys

for _p in (str(_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langsmith import uuid7

from eval_lab.agent import extract_tool_calls
from utils.models import judge_model


def make_run_agent(agent):
    """Target function: run `agent` on the query, return the answer and tool-call sequence."""
    def run_agent(inputs: dict) -> dict:
        config = {"configurable": {"thread_id": str(uuid7())}}
        result = agent.invoke({"messages": [{"role": "user", "content": inputs["query"]}]}, config=config)
        return {
            "answer": result["messages"][-1].text,
            "trajectory": extract_tool_calls(result["messages"]),
        }
    return run_agent


def exact_trajectory(outputs, reference_outputs):
    """Code evaluator: 1 if the tool-call sequence matches the reference exactly."""
    return {"key": "exact_trajectory",
            "score": int(outputs["trajectory"] == reference_outputs["trajectory"])}


def keyword_match(outputs, reference_outputs):
    """Code evaluator: fraction of reference keywords present in the answer."""
    words = reference_outputs.get("keywords", [])
    text = (outputs.get("answer") or "").lower()
    hit = sum(1 for w in words if w.lower() in text)
    return {"key": "keyword_match", "score": hit / len(words) if words else 1.0}


def judge_json(prompt: str) -> dict:
    """Ask the judge model for a JSON object. Never raises: a bad reply becomes score None."""
    try:
        text = judge_model.invoke(prompt).text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group(0))
    except Exception as exc:  # judge failures are recorded, not fatal
        return {"correct": None, "reason": f"judge failed: {type(exc).__name__}"}


ANSWER_JUDGE = (
    "You grade an order-support assistant.\n"
    "Customer asked: {query}\n"
    "Assistant answered: {answer}\n"
    "Reference keywords: {keywords}\n"
    "Is the answer accurate and does it handle the request correctly? "
    "Reply with only JSON: {{\"correct\": 0 or 1, \"reason\": \"one sentence\"}}"
)


def answer_quality(inputs, outputs, reference_outputs):
    """LLM-as-judge evaluator. The judge's rationale is stored as the feedback comment."""
    verdict = judge_json(ANSWER_JUDGE.format(
        query=inputs["query"], answer=outputs.get("answer"), keywords=reference_outputs.get("keywords")))
    return {"key": "answer_quality", "score": verdict.get("correct"), "comment": verdict.get("reason")}


EVALUATOR_VERSION = "answer_quality@v1+exact_trajectory@v1+keyword_match@v1"
