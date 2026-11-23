"""Reference solutions for the enrichment pieces you write in challenges 2 to 5.

Later notebooks import the earlier pieces from here, so a challenge never depends on you having
finished the one before it. The reference solutions shown in the notebooks are copied from this file.
"""

import json
from importlib.metadata import version

from opentelemetry import baggage, context
from opentelemetry.sdk.trace import SpanProcessor

import bank_agent
import bank_traffic
import otel_setup

DEPLOYMENT = {
    "environment": "uat",
    "service_name": "bank-support-agent",
    "cluster": "sg-aks-uat-01",
    "namespace": "genai-agents",
    "pod_name": "bank-agent-7d9f-xk2lp",
}


def resource_attributes():
    """Challenge 2: process-wide constants, attached to every span."""
    return {
        **DEPLOYMENT,
        "agent_name": "support-assistant",
        "agent_id": "bank-support-agent",
        "agent_version": "2026.10.1",
        "agent_framework": "langchain",
        "agent_framework_version": version("langchain"),
    }


class RequestContextProcessor(SpanProcessor):
    """Challenge 3: stamp per-request baggage onto every span in the trace."""

    def __init__(self):
        self.by_trace = {}

    def on_start(self, span, parent_context=None):
        bag = baggage.get_all(parent_context or context.get_current())
        trace_id = span.get_span_context().trace_id
        if bag:
            self.by_trace[trace_id] = dict(bag)
        meta = self.by_trace.get(trace_id)
        if meta:
            span.set_attributes(otel_setup.to_langsmith_attributes(meta))


def risk_metadata(question, channel):
    """Challenge 4: guardrail, PII and policy values for one turn."""
    pii = bank_agent.detect_pii(question)
    safety = bank_agent.content_safety(question)
    sensitivity = bank_agent.classify(question, pii)
    decision, ref = bank_agent.policy_decision(sensitivity, channel, safety)
    return {
        "data_sensitivity": sensitivity,
        "pii_detected": bool(pii),
        "pii_categories": ",".join(pii),
        "content_safety": safety,
        "guardrail_name": "input-guardrail",
        "guardrail_id": "grd-0042",
        "guardrail_version": "2.1.0",
        "guardrail_score": 0.0 if safety == "safe" else 0.97,
        "eval_criteria": "policy_compliance",
        "policy_decision": decision,
        "policy_decision_ref": ref,
        "hitl_escalated": decision == "escalate",
        "maker_checker_ref": "mc-pending" if decision == "escalate" else "",
        "blocked_response": decision == "deny",
        "fallback_used": False,
        "retry_count": 0,
    }


TOOL_REGISTRY = {
    "account_lookup": {"tool_id": "tool-acct-001", "tool_version": "2.3.1", "mcp_server": "account-mcp"},
}


def tool_metadata(name, tool_input, output, error, elapsed_ms):
    """Challenge 4: describe one tool or retrieval call."""
    if name == "search_kb":
        docs = (output or {}).get("documents", [])
        return {
            "knowledge_source": bank_agent.KNOWLEDGE_SOURCE,
            "retrieval_query": tool_input["query"],
            "retrieval_scores": [d["metadata"].get("score") for d in docs],
            "citations": [d["metadata"].get("source") for d in docs],
            "retrieval_latency_ms": elapsed_ms,
        }
    reg = TOOL_REGISTRY[name]
    import uuid

    return {
        "tool_id": reg["tool_id"],
        "tool_version": reg["tool_version"],
        "tool_call_id": str(uuid.uuid4()),
        "tool_outcome": "error" if error else "success",
        "mcp_server": reg["mcp_server"],
        "mcp_request": json.dumps(tool_input),
        "mcp_response": str(error) if error else json.dumps(output),
    }


def geography_attributes(conversation_id):
    """Challenge 5: where the customer is, as span attributes and tags."""
    geo = bank_traffic.GEO_REGISTRY.lookup(conversation_id)
    attrs = otel_setup.to_langsmith_attributes(geo)
    attrs["langsmith.span.tags"] = [f"region:{geo['customer_region']}", f"country:{geo['customer_country']}"]
    return attrs


class GeographyProcessor(RequestContextProcessor):
    """Challenge 5 (provided): the challenge 3 processor plus geography for the trace.

    `geography_fn(conversation_id)` returns span attributes; challenge 5 passes yours.
    """

    def __init__(self, geography_fn):
        super().__init__()
        self.geography_fn = geography_fn

    def on_start(self, span, parent_context=None):
        super().on_start(span, parent_context)
        meta = self.by_trace.get(span.get_span_context().trace_id) or {}
        if meta.get("conversation_id"):
            span.set_attributes(self.geography_fn(meta["conversation_id"]))


def per_request(conversation_id):
    """The baggage a real gateway would set from the session store and the request."""
    s = bank_traffic.SESSION_REGISTRY.lookup(conversation_id)
    return {**s, "caller_identity": "service:bank-support-agent"}


def ensure_traffic(project, minimum=12):
    """Run the fully enriched agent until `project` has `minimum` root runs, for later challenges.

    Uses the reference pieces from every earlier challenge. Returns the root count.
    """
    from langsmith import Client

    import enrich

    have = len([r for r in enrich.project_runs(Client(), project) if r.parent_run_id is None])
    if have >= minimum:
        return have
    otel_setup.configure(resource_attributes=resource_attributes(),
                         processors=[GeographyProcessor(geography_attributes)])
    agent = bank_agent.build_agent(risk_hook=risk_metadata, tool_hook=tool_metadata)
    questions = (bank_traffic.QUESTIONS * 2)[: max(minimum - have, 4) + 2]
    bank_traffic.run_turns(bank_agent, agent, questions, per_request=per_request)
    otel_setup.flush()
    return bank_traffic.wait_for_runs(project, minimum=minimum)
