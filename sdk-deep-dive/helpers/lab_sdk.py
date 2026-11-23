"""Plumbing for the SDK deep dive notebooks and checks.

Naming, trace seeding, redaction and run-tree helpers live here so the
notebooks stay about the SDK calls. Nothing here is hidden magic: every
function is short and readable.
"""
import pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parents[1]

import ast
import os
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# The run-query methods used in this track (list_runs, read_run) are deprecated
# in favour of client.runs.query(); they keep working until 2027-01-31.
# No module= filter: LangSmith warns with stacklevel>1, so the warning is attributed to the caller.
warnings.filterwarnings("ignore", message=r"(?is).*deprecat.*")
warnings.filterwarnings("ignore", message=r"(?is).*(list_runs|read_run|list_threads|session_id).*")
try:  # beta notices and LangSmith's own warning classes
    from langsmith._internal._beta_decorator import LangSmithBetaWarning
    from langsmith.utils import LangSmithWarning
    warnings.filterwarnings("ignore", category=LangSmithBetaWarning)
    warnings.filterwarnings("ignore", category=LangSmithWarning)
except ImportError:
    pass

for _p in (str(_ROOT), str(_ROOT / "helpers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EXPORT_DIR = str(_ROOT / "exports")


# ---------------------------------------------------------------- names
def sfx():
    return os.environ.get("WORKSHOP_SFX", "local")


def prod_project():
    return os.environ["LANGSMITH_PROJECT"] + "-prod"


def ds_traces():
    return f"sdk-from-traces-{sfx()}"


def ds_golden():
    return f"sdk-golden-{sfx()}"


def ds_uploaded():
    return f"sdk-uploaded-{sfx()}"


def queue_name():
    return f"sdk-review-{sfx()}"


def prompt_name():
    return f"order-support-{sfx()}"


def run_meta(run):
    """Metadata dict of a run (LangSmith stores it under run.extra['metadata'])."""
    return (getattr(run, "extra", None) or {}).get("metadata", {}) or {}


# ---------------------------------------------------------------- redaction
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def redact(text):
    """Reference redaction: emails and phone numbers."""
    text = EMAIL.sub("[EMAIL]", text)
    return PHONE.sub("[PHONE]", text)


# ---------------------------------------------------------------- run helpers
def tool_args(run):
    """Arguments passed to a tool run, as a dict, whichever way LangChain recorded them."""
    inputs = run.inputs or {}
    if "order_id" in inputs or "query" in inputs:
        return inputs
    raw = inputs.get("input")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return {}


def auto_score(run):
    """Process check for a root run: 0 on error, or on a refund without an eligibility check."""
    if run.error:
        return 0
    traj = (run.outputs or {}).get("trajectory", [])
    if "issue_refund" in traj and "check_refund_eligibility" not in traj:
        return 0
    return 1


def human_label(run):
    """Stand-in for reviewer labels in challenge 7: 1 = correct, with a reviewer-style comment."""
    traj = (run.outputs or {}).get("trajectory", [])
    if run.error:
        return 0, "Run errored; customer got no answer."
    if "issue_refund" in traj and "check_refund_eligibility" not in traj:
        return 0, "Refund issued without checking eligibility."
    return 1, "Handled correctly."


def roots(client, start=None):
    """Root runs of the seeded production project, in a stable order."""
    runs = list(client.list_runs(project_name=prod_project(), is_root=True, start_time=start))
    return sorted(runs, key=lambda r: (run_meta(r).get("release", ""), str((r.inputs or {}).get("query", "")), str(r.id)))


# ---------------------------------------------------------------- trace seeding
QUERIES = [
    ("refund", "Hi, I'm Dana (dana.reyes@example.com). Refund order A1001 please."),
    ("refund", "I want a refund for A1002, it's been a while. Reach me at dana.reyes@example.com"),
    ("refund", "Refund order A1004, contact lee.park@example.com"),
    ("refund", "Refund order A1003 now! Call +1 415-555-0134"),
    ("order-status", "Where is order A1003? My number is 415-555-0188"),
    ("order-status", "Can you check the status of order A1001?"),
    ("order-status", "Status of order A1005?"),
    ("faq", "What's your return policy?"),
    ("faq", "How long does standard shipping take?"),
    ("faq", "What warranty do your products have?"),
]

THREADS = [
    ["Hi, where is order A1001?", "Great, can I get a refund for it?"],
    ["What's your return policy?", "And how long does shipping take?"],
    ["What's the status of order A1002?", "Refund it please."],
]

RELEASES = {"v1.0": "SYSTEM_PROMPT_V1", "v1.1": "SYSTEM_PROMPT_V2"}
EXPECTED_ROOTS = len(QUERIES) * len(RELEASES) + sum(len(t) for t in THREADS)


def seed(client=None, force=False):
    """Send ~26 production-style traces (two releases, tags, threads, errors) to the prod project.

    Idempotent: does nothing if the project already holds the full set.
    """
    from langsmith import Client, traceable
    from langchain_core.tracers.langchain import wait_for_all_tracers

    from eval_lab import agent as agent_mod

    client = client or Client()
    project = prod_project()
    if not force:
        try:
            have = len(list(client.list_runs(project_name=project, is_root=True, select=["id"])))
        except Exception:
            have = 0
        if have >= EXPECTED_ROOTS:
            print(f"{project}: already seeded ({have} root runs)")
            return

    @traceable(name="support_agent", run_type="chain", client=client, project_name=project)
    def support_turn(query: str, history: list, release: str) -> dict:
        prompt = getattr(agent_mod, RELEASES[release])
        agent = agent_mod.build_agent(prompt)
        messages = list(history) + [{"role": "user", "content": query}]
        result = agent.invoke({"messages": messages})
        return {"answer": result["messages"][-1].text,
                "trajectory": agent_mod.extract_tool_calls(result["messages"])}

    bedrock_errors = []

    def one(job):
        query, history, release, tags, meta = job
        try:
            return support_turn(query, history, release,
                                langsmith_extra={"tags": tags, "metadata": {**meta, "release": release}})
        except Exception as exc:
            if "AccessDenied" in str(exc) or "ValidationException" in str(exc):
                bedrock_errors.append(str(exc))  # model access problem, not a deliberate failure
            return None  # the failure is the point: it is recorded on the trace

    jobs = []
    for release in RELEASES:
        for i, (segment, query) in enumerate(QUERIES):
            env = "staging" if i % 5 == 4 else "prod"
            jobs.append((query, [], release, [env, segment], {"env": env, "segment": segment}))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, jobs))

    # Threads run turn by turn so each turn sees the previous answers.
    def thread(n_turns):
        n, turns = n_turns
        thread_id = f"thread-{sfx()}-{n}"
        history = []
        for query in turns:
            out = one((query, history, "v1.1", ["prod", "thread"], {"env": "prod", "thread_id": thread_id}))
            history += [{"role": "user", "content": query},
                        {"role": "assistant", "content": (out or {}).get("answer", "")}]

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(thread, enumerate(THREADS, 1)))

    wait_for_all_tracers()
    if bedrock_errors:
        raise RuntimeError(
            f"{len(bedrock_errors)} agent calls were rejected by Bedrock, so the seeded traces are error traces. "
            f"First error: {bedrock_errors[0][:300]}")
    client.flush()
    print(f"{project}: seeded {len(jobs) + sum(len(t) for t in THREADS)} traces "
          f"(releases {', '.join(RELEASES)}; tags env/segment; 3 threads; some errors)")


# ---------------------------------------------------------------- shared setup for notebooks
def start_of_today():
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------- experiments
def experiments(client, dataset_id, prefix=""):
    """Experiments on a dataset whose name starts with `prefix`, oldest first."""
    found = [p for p in client.list_projects(reference_dataset_id=dataset_id) if p.name.startswith(prefix)]
    return sorted(found, key=lambda p: p.start_time)


def experiment_scores(client, project_name):
    """(root runs, {feedback key: average score}) for one experiment."""
    from collections import defaultdict
    runs = list(client.list_runs(project_name=project_name, is_root=True, select=["id"]))
    totals, counts = defaultdict(float), defaultdict(int)
    if runs:
        for fb in client.list_feedback(run_ids=[r.id for r in runs]):
            if fb.score is not None:
                totals[fb.key] += fb.score
                counts[fb.key] += 1
    return len(runs), {k: totals[k] / counts[k] for k in totals}


def compare(prefixes, dataset_name=None):
    """Print one row per experiment: runs and average score per feedback key."""
    from langsmith import Client
    client = Client()
    ds = client.read_dataset(dataset_name=dataset_name or ds_golden())
    rows = []
    for p in experiments(client, ds.id):
        if any(p.name.startswith(x) for x in prefixes):
            n, scores = experiment_scores(client, p.name)
            rows.append((p.name, n, scores))
    if not rows:
        print("No matching experiments yet.")
        return
    cols = sorted({k for _, _, s in rows for k in s})
    width = max(len(r[0]) for r in rows)
    print(f"{'experiment':<{width}}  runs  " + "  ".join(f"{c:>16}" for c in cols))
    for name, n, scores in rows:
        cells = "  ".join(f"{scores[c]:>16.2f}" if c in scores else f"{'-':>16}" for c in cols)
        print(f"{name:<{width}}  {n:>4}  {cells}")


# ---------------------------------------------------------------- feedback helpers
def feedback_when_ready(client, run_id, retries=8, **kwargs):
    """create_feedback, retrying while a freshly uploaded run is still being ingested."""
    import time
    last = None
    for _ in range(retries):
        try:
            return client.create_feedback(run_id=run_id, **kwargs)
        except Exception as exc:  # run not ingested yet
            last = exc
            time.sleep(3)
    raise last


def wait_for_feedback(client, key, run_ids, expected, timeout=90):
    """Poll until `expected` feedback rows with `key` are readable (writes are asynchronous)."""
    import time
    end = time.time() + timeout
    seen = {}
    while time.time() < end:
        seen = {f.run_id: f.score for f in client.list_feedback(run_ids=run_ids, feedback_key=[key])}
        if len(seen) >= expected:
            break
        time.sleep(3)
    return seen
