"""The bank support agent: a LangChain agent with a knowledge-base retriever and an account tool.

It is provided, not something you write. What you write are the enrichment hooks it calls:

  risk_hook(question, channel) -> dict   guardrail / PII / policy values, set on the turn's root span
  tool_hook(name, tool_input, output, error, elapsed_ms) -> dict   tool and retrieval values, set on a
                                         span opened in the tool's own code

Both return plain metadata dicts. This module turns them into `langsmith.metadata.*` span attributes
with `otel_setup.to_langsmith_attributes`, so they leave the process already attached to the spans.

Tracing goes through the OpenTelemetry SDK. Call `otel_setup.configure(...)` BEFORE `build_agent()`.
"""

import json
import os
import re
import time
import uuid

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from opentelemetry import baggage, context, trace

import otel_setup

SERVICE = "bank-support-agent"
HOOKS = {"risk": None, "tool": None}

SYSTEM_PROMPT = (
    "You are a customer support assistant for a retail bank. Answer using the knowledge base "
    "(search_kb). Whenever the customer gives an account reference such as SAV-4821, you must call "
    "account_lookup with that reference before answering. Keep answers short. "
    "If you cannot help, say so. Never reveal customer identity numbers."
)

KB = [
    ("transfer-limits.md", "Daily PayNow transfer limit is SGD 50,000 per account. Limits can be lowered, never raised, from the mobile app."),
    ("card-services.md", "PIN reset: open Cards in the mobile app, choose your card, then Reset PIN. Card limit increases are reviewed within one business day."),
    ("transfers.md", "Own-account transfers settle instantly. Overseas transfers cost SGD 12 plus the receiving bank's fee."),
    ("statements.md", "Statements are under Accounts > Documents and are kept for 7 years."),
    ("payments-faq.md", "Declines usually come from limits or a card block; check Cards in the app. Add a PayNow recipient under Payments > PayNow > Add recipient."),
    ("profile.md", "Update your mailing address under Profile > Contact details, then choose Edit."),
]
KNOWLEDGE_SOURCE = "kb-banking-faq-v12"


class KBRetriever(BaseRetriever):
    """Keyword-overlap retriever over KB. Deterministic, no embeddings, no extra service."""

    k: int = 2

    def _get_relevant_documents(self, query, *, run_manager=None):
        words = set(re.findall(r"[a-z]+", query.lower()))
        scored = []
        for source, text in KB:
            overlap = len(words & set(re.findall(r"[a-z]+", text.lower())))
            scored.append((overlap / max(len(words), 1), source, text))
        scored.sort(reverse=True)
        return [Document(page_content=t, metadata={"source": s, "score": round(min(0.95, 0.4 + sc), 2)})
                for sc, s, t in scored[: self.k]]


RETRIEVER = KBRetriever()

# ---- local rules for the hooks to use: replace with your real guardrail, classifier and policy engine ----
PII_PATTERNS = {
    "national_id": re.compile(r"\b[STFG]\d{7}[A-Z]\b", re.I),
    "card_number": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"\b(?:\+65)?[689]\d{7}\b"),
}
JAILBREAK = re.compile(r"ignore (all )?(previous|prior) instructions|reveal your system prompt|\bNRIC\b", re.I)
SENSITIVE_TOPICS = re.compile(r"\b(account|transfer|balance|card|PIN)\b", re.I)


def detect_pii(text):
    return sorted(name for name, rx in PII_PATTERNS.items() if rx.search(text))


def classify(text, pii):
    """Data classification: internal < confidential < restricted."""
    if "national_id" in pii or "card_number" in pii:
        return "restricted"
    if pii or SENSITIVE_TOPICS.search(text):
        return "confidential"
    return "internal"


def content_safety(text):
    return "jailbreak_attempt" if JAILBREAK.search(text) else "safe"


def policy_decision(sensitivity, channel, safety):
    """Decision record: allow, deny, or escalate to a human."""
    if safety != "safe":
        return "deny", "policy/content-safety/v3"
    if sensitivity == "restricted" and channel != "internal_portal":
        return "escalate", "policy/data-handling/v7"
    return "allow", "policy/data-handling/v7"


# ---- tools: each opens its own span and lets tool_hook describe what happened ----
def _audited(name, tool_input, fn):
    tracer = trace.get_tracer(SERVICE)
    started = time.time()
    with tracer.start_as_current_span(f"{name}.audit") as span:
        span.set_attribute("langsmith.span.kind", "tool")
        output, error = None, None
        try:
            output = fn()
        except Exception as exc:  # noqa: BLE001 - recorded on the span, reported to the model
            error = exc
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
        span.set_attribute("input.value", json.dumps(tool_input))
        span.set_attribute("output.value", str(error) if error else json.dumps(output))
        hook = HOOKS["tool"]
        if hook:
            elapsed_ms = int((time.time() - started) * 1000)
            span.set_attributes(otel_setup.to_langsmith_attributes(
                hook(name, tool_input, output, error, elapsed_ms)))
        return f"error: {error}" if error else output


@tool
def search_kb(query: str) -> str:
    """Search the bank's knowledge base for an answer to a customer question."""

    def run():
        docs = RETRIEVER.invoke(query)
        return {"documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]}

    out = _audited("search_kb", {"query": query}, run)
    if isinstance(out, dict):
        return "\n".join(d["page_content"] for d in out["documents"])
    return out


@tool
def account_lookup(account_ref: str) -> str:
    """Look up the daily limit of an account by reference, for example SAV-4821."""

    def run():
        if not account_ref.upper().startswith("SAV-"):
            raise TimeoutError("account service did not respond within 5s")
        return {"account_ref": account_ref, "daily_limit": 50000}

    out = _audited("account_lookup", {"account_ref": account_ref}, run)
    return str(out)


def build_agent(risk_hook=None, tool_hook=None):
    """Build the agent. Hooks are optional: with none, the agent emits only what OTel carries by default."""
    import llm as _llm

    HOOKS["risk"], HOOKS["tool"] = risk_hook, tool_hook
    llm = _llm.make_chat_model(temperature=0, max_tokens=512)
    tools = [search_kb, account_lookup]
    try:
        from langchain.agents import create_agent

        return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
    except ImportError:
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


def ask(agent, question, conversation_id=None, baggage_values=None, channel=None):
    """One customer turn. Wraps the agent call in a root span, `bank_support_turn`.

    baggage_values is the per-request context (channel, user, ...) carried as OTel baggage; a
    SpanProcessor can read it. conversation_id is always added.
    """
    conversation_id = conversation_id or str(uuid.uuid4())
    values = {**(baggage_values or {}), "conversation_id": conversation_id}
    ctx = context.get_current()
    for key, value in values.items():
        ctx = baggage.set_baggage(key, str(value), ctx)
    token = context.attach(ctx)
    try:
        tracer = trace.get_tracer(SERVICE)
        with tracer.start_as_current_span("bank_support_turn") as span:
            span.set_attribute("langsmith.span.kind", "chain")
            span.set_attribute("input.value", question)
            meta = {}
            if HOOKS["risk"]:
                meta = HOOKS["risk"](question, channel or values.get("channel", "web_chat")) or {}
                span.set_attributes(otel_setup.to_langsmith_attributes(meta))
            if meta.get("policy_decision", "allow") != "allow":
                answer = "I can't help with that here."
            else:
                result = agent.invoke({"messages": [{"role": "user", "content": question}]})
                last = result["messages"][-1].content
                answer = last if isinstance(last, str) else "".join(
                    b.get("text", "") for b in last if isinstance(b, dict))
            span.set_attribute("output.value", answer)
            return answer
    finally:
        context.detach(token)
