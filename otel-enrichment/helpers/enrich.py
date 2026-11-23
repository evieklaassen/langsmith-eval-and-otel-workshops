"""Read helpers for traces that are already in LangSmith.

Enrichment itself happens BEFORE export (see otel_setup, bank_agent and enrichment_ref). This module only
reads: it lists runs, pulls metadata and text out of them, and prints a trace tree.
"""

import json
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import islice

# list_runs() is deprecated (removal after 2027-01-31) in favour of the async-only
# client.runs.query(), which also changes the Run fields the notebooks read. The sync
# call still works, so keep learner output clean until the track is migrated.
warnings.filterwarnings("ignore", message=r"list_runs\(\) is deprecated")


def project_runs(client, project, hours=24, limit=500, **filters):
    """All runs in the project from the last `hours` hours (newest first)."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    # The API caps a page at 100; list_runs paginates, so cap the total client-side.
    runs = client.list_runs(project_name=project, start_time=since, **filters)
    return list(islice(runs, limit))


def meta_of(run):
    return dict((getattr(run, "extra", None) or {}).get("metadata") or {})


RESOURCE_PREFIX = "otel.resource."


def meta_get(run, key, default=None):
    """A metadata value by its plain name. LangSmith stores OTel resource attributes under
    `otel.resource.<key>`, so that spelling counts too."""
    m = meta_of(run)
    return m[key] if key in m else m.get(RESOURCE_PREFIX + key, default)


def has_meta(run, key):
    m = meta_of(run)
    return key in m or (RESOURCE_PREFIX + key) in m


def agent_runs(runs):
    """Runs in traces that contain the agent's `bank_support_turn` root span.

    If the agent's root span is not there (the exporter named it differently), every named run
    counts, so a naming difference never hides your traces.
    """
    traces = {r.trace_id for r in runs if r.name == "bank_support_turn"}
    named = [r for r in runs if r.trace_id in traces]
    return named or [r for r in runs if r.name]


def _first_text(d):
    for v in (d or {}).values():
        if isinstance(v, str) and v:
            return v
    return ""


def _json_values(d):
    for v in (d or {}).values():
        if isinstance(v, str) and v[:1] in "{[":
            try:
                yield json.loads(v)
            except ValueError:
                pass
        elif isinstance(v, (dict, list)):
            yield v


def question_of(run):
    """The user's question, from LangSmith-style messages or an OTel input.value."""
    inputs = run.inputs or {}
    msgs = inputs.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict):
        return str(msgs[-1].get("content", ""))
    return inputs.get("query") or inputs.get("question") or _first_text(inputs)


def answer_of(run):
    out = run.outputs or {}
    return out.get("answer") or _first_text(out)


def documents_of(run):
    """Retrieved documents (list of {page_content, metadata}) from a retriever or tool run."""
    out = run.outputs or {}
    if isinstance(out.get("documents"), list):
        return out["documents"]
    for v in _json_values(out):
        if isinstance(v, dict) and isinstance(v.get("documents"), list):
            return v["documents"]
        if isinstance(v, list):
            return v
    return []


def show_trace(runs, trace_id=None):
    """Print one trace as an indented tree: name, type, tags, metadata keys."""
    by_parent = defaultdict(list)
    for r in runs:
        by_parent[r.parent_run_id].append(r)
    roots = [r for r in runs if r.parent_run_id is None and (trace_id is None or r.trace_id == trace_id)]
    if not roots:
        print("No runs found.")
        return

    def walk(run, depth):
        keys = ", ".join(sorted(meta_of(run))) or "-"
        print(f"{'  ' * depth}{run.name}  [{run.run_type}]  tags={run.tags or []}")
        print(f"{'  ' * depth}  metadata keys: {keys}")
        for child in by_parent.get(run.id, []):
            walk(child, depth + 1)

    walk(roots[0], 0)
