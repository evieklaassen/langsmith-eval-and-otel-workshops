"""Quality, cost and latency of an experiment, by segment.

Shared by the challenge 5 notebook and the challenge 8 gate, so both read an
experiment the same way.

PRICES is an ASSUMPTION for the lab (USD per 1M tokens, approximate list prices).
Replace it with your contract rates before using any cost figure for a real decision.
"""

import math
from collections import defaultdict

from utils.models import BASELINE_ID, BASELINE_PRICE, CANDIDATE_ID, CANDIDATE_PRICE

KEYS = ("exact_trajectory", "keyword_match", "answer_quality")

PRICES = {  # ASSUMPTION: USD per 1M tokens; set in utils/models.py (override with MODEL_*_PRICE in .env)
    BASELINE_ID: BASELINE_PRICE,
    CANDIDATE_ID: CANDIDATE_PRICE,
}


def run_cost(run, model_id):
    """Reference cost of one run in USD, from its token counts and the price table."""
    price = PRICES[model_id]
    return ((run.prompt_tokens or 0) * price["input"] + (run.completion_tokens or 0) * price["output"]) / 1_000_000


def percentile(values, q):
    """Nearest-rank percentile (q in 0..100). None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def policy_violation(run):
    """1 if the agent issued a refund without checking eligibility (the safety rule)."""
    traj = (run.outputs or {}).get("trajectory", []) or []
    return int("issue_refund" in traj and "check_refund_eligibility" not in traj)


def experiment_rows(client, experiment_name, model_id, example_segments, cost_fn=run_cost, wait=60):
    """One dict per example run: segment, quality scores, cost, latency, error, violation.

    Feedback is written asynchronously, so poll until every run has all score keys (or `wait` runs out).
    """
    import time
    end = time.time() + wait
    while True:
        runs = list(client.list_runs(project_name=experiment_name, is_root=True))
        scores = defaultdict(dict)
        for fb in client.list_feedback(run_ids=[r.id for r in runs]):
            if fb.key in KEYS and fb.score is not None:
                scores[fb.run_id][fb.key] = fb.score
        complete = runs and all(len(scores[r.id]) == len(KEYS) or r.error for r in runs)
        if complete or time.time() > end:
            break
        time.sleep(5)
    rows = []
    for r in runs:
        q = [scores[r.id][k] for k in KEYS if k in scores[r.id]]
        rows.append({
            "segment": example_segments.get(r.reference_example_id, "?"),
            "scores": dict(scores[r.id]),
            "quality": sum(q) / len(q) if q else None,
            "cost": cost_fn(r, model_id),
            "latency_s": (r.end_time - r.start_time).total_seconds() if r.end_time and r.start_time else None,
            "error": bool(r.error),
            "violation": policy_violation(r),
        })
    return rows


def summarize(rows):
    """Overall and per-segment: quality, each score key, cost per example, quality per dollar, latency p50/p95, error rate."""
    def one(rs):
        cost = _mean([r["cost"] for r in rs])
        quality = _mean([r["quality"] for r in rs])
        lat = [r["latency_s"] for r in rs if r["latency_s"] is not None]
        out = {
            "n": len(rs),
            "quality": quality,
            "cost_per_example": cost,
            "quality_per_dollar": (quality / cost) if quality is not None and cost else None,
            "latency_p50": percentile(lat, 50),
            "latency_p95": percentile(lat, 95),
            "error_rate": sum(r["error"] for r in rs) / len(rs) if rs else None,
            "policy_violations": sum(r["violation"] for r in rs),
        }
        for k in KEYS:
            out[k] = _mean([r["scores"].get(k) for r in rs])
        return out

    by_seg = defaultdict(list)
    for r in rows:
        by_seg[r["segment"]].append(r)
    return {"overall": one(rows), "segments": {s: one(rs) for s, rs in sorted(by_seg.items())}}


def example_segments(client, dataset_name):
    return {e.id: (e.metadata or {}).get("segment", "?") for e in client.list_examples(dataset_name=dataset_name)}


def latest_experiment(client, dataset_id, prefix):
    """Newest experiment on the dataset whose name starts with `prefix`, or None."""
    found = [p for p in client.list_projects(reference_dataset_id=dataset_id) if p.name.startswith(prefix)]
    return max(found, key=lambda p: p.start_time).name if found else None
