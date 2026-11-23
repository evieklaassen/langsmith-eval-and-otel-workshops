"""Measure which Phase 2 attributes are present in the learner's LangSmith traces.

The 83 attributes come from phase2_requirements.csv (same order as the source file).
Each row is paired with:
  stage    - "native": expected in the spans LangChain's OTel integration sends by default;
             an int: the challenge where you add it (2 resource attributes, 3 SpanProcessor
             and baggage, 4 span attributes in tool code, 5 geography, 6 evaluator, 7 annotation
             queue);
             "ref": reference only, not built hands-on in this track.
  where    - where the attribute lives in LangSmith.
  detector - how presence is measured, or None for reference-only rows.
             ("field", name)    a Run field is populated (trace_id, tags, ...)
             ("meta", key)      extra.metadata has the key on some run
             ("name", prefix)   some run name starts with prefix
             ("runtype", type)  some run has this run_type
             ("feedback", key)  feedback with this key exists
Presence is MEASURED against your project, never assumed: "native" is what LangChain's OTel
integration should send, and a gap in your traces is a finding.
"""

import csv
import json
import os
from pathlib import Path

import pandas as pd

from enrich import has_meta, meta_of

CSV_PATH = Path(__file__).with_name("phase2_requirements.csv")

# One entry per CSV row, in order: (stage, where, detector)
ROWS = [
    ('native', 'run name bank_support_turn, run_type chain (root)', ('name', 'bank_support_turn')),
    ('ref', 'metadata.plan_steps', None),
    ('ref', 'metadata.branch_taken', None),
    ('native', 'run.trace_id', ('field', 'trace_id')),
    ('native', 'run.dotted_order', ('field', 'dotted_order')),
    (3, 'metadata.conversation_id (thread grouping)', ('meta', 'conversation_id')),
    (2, 'metadata.environment', ('meta', 'environment')),
    (2, 'metadata.service_name', ('meta', 'service_name')),
    (2, 'metadata.cluster + metadata.namespace', ('meta', 'namespace')),
    (2, 'metadata.pod_name', ('meta', 'pod_name')),
    (6, 'feedback key judge_score (evaluator / LLM-judge result)', ('feedback', 'judge_score')),
    (3, 'metadata.channel', ('meta', 'channel')),
    ('ref', 'metadata.detected_language', None),
    (4, 'metadata.fallback_used', ('meta', 'fallback_used')),
    ('native', 'metadata.ls_provider', ('meta', 'ls_provider')),
    ('native', 'metadata.ls_model_name', ('meta', 'ls_model_name')),
    ('ref', 'metadata.response_model', None),
    ('ref', 'metadata.ls_temperature', None),
    ('native', 'run.inputs (LLM run)', ('field', 'inputs')),
    ('native', 'run.outputs (LLM run)', ('field', 'outputs')),
    ('ref', 'thinking content block in run.outputs', None),
    ('ref', 'metadata.ls_stop', None),
    ('ref', 'metadata.finish_reason', None),
    (4, 'metadata.citations', ('meta', 'citations')),
    ('ref', 'metadata.ttft_ms', None),
    ('ref', 'metadata.model_hosting', None),
    ('native', 'run.inputs.query', ('field', 'inputs')),
    ('native', 'tool run metadata.tool_inputs', ('meta', 'tool_inputs')),
    ('native', 'tool run outputs', ('meta', 'tool_outputs')),
    (4, 'metadata.tool_id + metadata.tool_version', ('meta', 'tool_version')),
    (4, 'metadata.tool_call_id', ('meta', 'tool_call_id')),
    ('native', 'tool run metadata.tool_inputs', ('meta', 'tool_inputs')),
    ('native', 'tool run metadata.tool_outputs', ('meta', 'tool_outputs')),
    ('native', 'tool run metadata.time_cost', ('meta', 'time_cost')),
    (4, 'metadata.tool_outcome', ('meta', 'tool_outcome')),
    ('native', 'retriever run outputs.documents[n].page_content', ('runtype', 'retriever')),
    (4, 'metadata.retrieval_query', ('meta', 'retrieval_query')),
    (4, 'metadata.retrieval_scores (+ outputs.documents)', ('meta', 'retrieval_scores')),
    (4, 'metadata.knowledge_source', ('meta', 'knowledge_source')),
    (4, 'metadata.data_sensitivity', ('meta', 'data_sensitivity')),
    (4, 'metadata.retrieval_latency_ms', ('meta', 'retrieval_latency_ms')),
    (4, 'metadata.mcp_server + mcp_request + mcp_response', ('meta', 'mcp_response')),
    (4, 'metadata.guardrail_score', ('meta', 'guardrail_score')),
    (4, 'metadata.guardrail_name / guardrail_id / guardrail_version', ('meta', 'guardrail_version')),
    (4, 'metadata.eval_criteria', ('meta', 'eval_criteria')),
    (4, 'metadata.policy_decision + policy_decision_ref', ('meta', 'policy_decision_ref')),
    (4, 'metadata.content_safety', ('meta', 'content_safety')),
    (4, 'metadata.pii_detected + pii_categories', ('meta', 'pii_detected')),
    (2, 'metadata.agent_name', ('meta', 'agent_name')),
    (2, 'metadata.agent_id + agent_version', ('meta', 'agent_version')),
    (2, 'metadata.agent_framework', ('meta', 'agent_framework')),
    ('ref', 'metadata.memory_retrieval', None),
    ('ref', 'feedback key user_score (end-user rating from your app)', None),
    (7, 'feedback key human_verdict (annotation queue)', ('feedback', 'human_verdict')),
    ('native', 'run.prompt_tokens', ('field', 'prompt_tokens')),
    ('native', 'run.completion_tokens', ('field', 'completion_tokens')),
    ('native', 'run.total_tokens', ('field', 'total_tokens')),
    ('ref', 'usage_metadata.output_token_details.reasoning', None),
    ('native', 'run.inputs (LLM run)', ('field', 'inputs')),
    ('native', 'run.outputs (LLM run)', ('field', 'outputs')),
    ('ref', 'system message inside run.inputs.messages', None),
    ('ref', 'prompt / instructions in run.inputs', None),
    ('native', 'run.start_time + run.end_time', ('field', 'end_time')),
    ('ref', 'metadata.tools_available', None),
    ('native', 'tool run name', ('field', 'name')),
    (4, 'metadata.retry_count', ('meta', 'retry_count')),
    ('native', 'run.name', ('field', 'name')),
    ('native', 'run.run_type', ('field', 'run_type')),
    ('native', 'run.id', ('field', 'id')),
    ('native', 'run.parent_run_id', ('field', 'parent_run_id')),
    ('native', 'run.run_type', ('field', 'run_type')),
    ('native', 'run.start_time', ('field', 'start_time')),
    (4, 'metadata.caller_identity', ('meta', 'caller_identity')),
    ('native', 'run.status', ('field', 'status')),
    ('native', 'run.error', ('field', 'error')),
    ('native', 'run.status', ('field', 'status')),
    ('ref', 'error text (stack trace only if the app captures it)', None),
    (4, 'metadata.blocked_response', ('meta', 'blocked_response')),
    (4, 'metadata.hitl_escalated', ('meta', 'hitl_escalated')),
    (4, 'metadata.maker_checker_ref', ('meta', 'maker_checker_ref')),
    ('ref', 'metadata.input_modality', None),
    ('ref', 'metadata.attachment_refs', None),
    (3, 'metadata.user_name', ('meta', 'user_name')),
]


def load():
    """The CSV joined with the mapping above, as a DataFrame."""
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(ROWS):
        raise SystemExit(f"coverage: CSV has {len(rows)} rows but the mapping has {len(ROWS)}")
    out = []
    for i, (row, (stage, where, det)) in enumerate(zip(rows, ROWS), start=1):
        out.append({
            "#": i,
            "attribute": row["Attribute Functional Name"],
            "csv_source": row["Generated By?"],
            "stage": stage,
            "where": where,
            "detector": det,
        })
    return pd.DataFrame(out)


def _present(det, runs, feedback_keys):
    kind, arg = det
    if kind == "field":
        return any(getattr(r, arg, None) not in (None, "", {}, []) for r in runs)
    if kind == "meta":
        return any(has_meta(r, arg) for r in runs)
    if kind == "name":
        return any((r.name or "").startswith(arg) for r in runs)
    if kind == "runtype":
        return any(r.run_type == arg for r in runs)
    if kind == "feedback":
        return arg in feedback_keys
    return False


def feedback_keys(client, runs):
    ordered = sorted(runs, key=lambda r: r.parent_run_id is not None)  # roots first: that is where feedback lives
    ids = [r.id for r in ordered][:200]
    if not ids:
        return set()
    try:
        return {f.key for f in client.list_feedback(run_ids=ids)}
    except Exception:  # noqa: BLE001
        return set()


def gap_report(client, project, runs=None, write=False):
    """Presence of every attribute in this project. Returns a DataFrame."""
    from enrich import project_runs

    runs = runs if runs is not None else project_runs(client, project)
    fkeys = feedback_keys(client, runs)
    df = load()

    def status(row):
        if row["detector"] is None:
            return "reference"
        return "present" if _present(row["detector"], runs, fkeys) else "missing"

    df["status"] = df.apply(status, axis=1)
    if write:
        summary = {
            "total_attributes": int(len(df)),
            "runs_examined": len(runs),
            "present": int((df["status"] == "present").sum()),
            "missing": int((df["status"] == "missing").sum()),
            "reference": int((df["status"] == "reference").sum()),
        }
        path = Path("/tmp/.gap-report.json")
        try:
            path.write_text(json.dumps(summary))
            os.chmod(path, 0o666)
        except PermissionError:
            pass
    return df.drop(columns=["detector"])


def summary(df):
    """Counts by stage and status."""
    d = df.copy()
    d["stage"] = d["stage"].astype(str)
    return d.pivot_table(index="stage", columns="status", values="#", aggfunc="count", fill_value=0)


def hands_on_coverage(df):
    """(present, total, missing attribute names) for the hands-on attributes (stages 2-7)."""
    hands_on = df[df["stage"].apply(lambda s: isinstance(s, int))]
    missing = hands_on[hands_on["status"] != "present"]
    return len(hands_on) - len(missing), len(hands_on), list(missing["attribute"])


def reference_table(df):
    """The attributes this track does not build hands-on."""
    ref = df[df["stage"] == "ref"][["#", "attribute", "csv_source", "where"]]
    return ref.rename(columns={"where": "suggested location in LangSmith"})
