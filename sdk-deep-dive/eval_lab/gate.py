"""Gated migration (challenge 8): decide whether a candidate model may replace the baseline.

    python -m eval_lab.gate            # exit 0 = PASS or PASS WITH CONDITIONS, 1 = BLOCKED

`run(decide)` is what the notebook calls with the learner's own decision rule. It REUSES
the baseline and candidate experiments from challenge 5 (newest `baseline*` and
`candidate*` experiments on the golden dataset), and only runs them if they are missing.

Three outcomes: PASS, PASS WITH CONDITIONS (ship, with named follow-ups), BLOCKED.
Thresholds below are TEMPLATE values: set them per use case in step 1 of the framework.
"""
import pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parents[1]

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

for _p in (str(_ROOT), str(_ROOT / "helpers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(str(_ROOT / ".env"), override=False)

from langsmith import Client

import lab_sdk
from eval_lab import metrics
from eval_lab.agent import SYSTEM_PROMPT_V2, build_agent
from eval_lab.evaluators import (EVALUATOR_VERSION, answer_quality, exact_trajectory,
                                 keyword_match, make_run_agent)
from utils.models import BASELINE_ID, CANDIDATE_ID, candidate_model, model

KEYS = metrics.KEYS
OUTCOMES = ("PASS", "PASS WITH CONDITIONS", "BLOCKED")

# Template thresholds (placeholders; set per use case).
THRESHOLDS = {
    "key_floor": 0.5,            # block: any score key below this
    "key_drop_block": 0.15,      # block: a score key drops more than this vs baseline
    "key_drop_warn": 0.075,      # condition: a score key drops more than this
    "segment_drop_block": 0.20,  # block: any segment's quality drops more than this
    "segment_drop_warn": 0.10,   # condition: any segment's quality drops more than this
    "cost_rise_pct": 50.0,       # cost per example rises more than this percent...
    "min_quality_gain": 0.02,    # ...block if quality gain is below this, else a condition
    "latency_p95_budget_s": 20.0,  # block: candidate p95 over this
    "latency_rise_warn": 1.5,    # condition: candidate p95 over this multiple of baseline
}


def default_decide(base, cand, t=THRESHOLDS):
    """Reference rule. `base` and `cand` are `metrics.summarize()` results.

    Returns (outcome, blocks, conditions): BLOCKED if there are blocks, PASS WITH CONDITIONS
    if only conditions, else PASS.
    """
    blocks, conditions = [], []
    bo, co = base["overall"], cand["overall"]

    for key in KEYS:
        if bo.get(key) is None or co.get(key) is None:
            blocks.append(f"{key}: missing score")
        elif co[key] < t["key_floor"]:
            blocks.append(f"{key}: {co[key]:.2f} is below the floor {t['key_floor']}")
        elif bo[key] - co[key] > t["key_drop_block"]:
            blocks.append(f"{key}: dropped {bo[key]:.2f} -> {co[key]:.2f}")
        elif bo[key] - co[key] > t["key_drop_warn"]:
            conditions.append(f"{key}: dropped {bo[key]:.2f} -> {co[key]:.2f}; review before full rollout")

    for seg, b in base["segments"].items():
        c = cand["segments"].get(seg)
        if c is None or b["quality"] is None or c["quality"] is None:
            conditions.append(f"segment {seg}: not measured for the candidate")
            continue
        drop = b["quality"] - c["quality"]
        if drop > t["segment_drop_block"]:
            blocks.append(f"segment {seg}: quality dropped {b['quality']:.2f} -> {c['quality']:.2f}")
        elif drop > t["segment_drop_warn"]:
            conditions.append(f"segment {seg}: quality dropped {b['quality']:.2f} -> {c['quality']:.2f}; monitor this segment")

    if co["policy_violations"] > bo["policy_violations"]:
        blocks.append(f"safety: {co['policy_violations']} refunds without an eligibility check (baseline {bo['policy_violations']})")

    if bo["cost_per_example"] and co["cost_per_example"] is not None:
        rise = (co["cost_per_example"] / bo["cost_per_example"] - 1) * 100
        gain = (co["quality"] or 0) - (bo["quality"] or 0)
        if rise > t["cost_rise_pct"]:
            msg = f"cost per example up {rise:.0f}% for a quality change of {gain:+.2f}"
            (blocks if gain < t["min_quality_gain"] else conditions).append(msg + ("" if gain < t["min_quality_gain"] else "; confirm the budget"))

    p95, base_p95 = co["latency_p95"], bo["latency_p95"]
    if p95 is not None:
        if p95 > t["latency_p95_budget_s"]:
            blocks.append(f"latency p95 {p95:.1f}s is over the {t['latency_p95_budget_s']:.0f}s budget")
        elif base_p95 and p95 > base_p95 * t["latency_rise_warn"]:
            conditions.append(f"latency p95 {base_p95:.1f}s -> {p95:.1f}s")

    outcome = "BLOCKED" if blocks else ("PASS WITH CONDITIONS" if conditions else "PASS")
    return outcome, blocks, conditions


def _normalize(result):
    """Accept (outcome, blocks, conditions) from a learner's decide(), or the old (passed, reasons) shape."""
    if len(result) == 3:
        return result
    passed, reasons = result
    return ("PASS" if passed else "BLOCKED"), ([] if passed else list(reasons)), []


def run_one(client, label, llm, model_id):
    results = client.evaluate(
        make_run_agent(build_agent(SYSTEM_PROMPT_V2, llm=llm)),
        data=client.list_examples(dataset_name=lab_sdk.ds_golden(), splits=["test"]),
        evaluators=[exact_trajectory, keyword_match, answer_quality],
        experiment_prefix=label,
        metadata={"model": model_id, "prompt": "v2", "evaluators": EVALUATOR_VERSION},
        max_concurrency=4,
    )
    return results.experiment_name


def experiments(client):
    """Names of the baseline and candidate experiments: reuse challenge 5's, run them if missing."""
    ds = client.read_dataset(dataset_name=lab_sdk.ds_golden())
    base = metrics.latest_experiment(client, ds.id, "baseline")
    cand = metrics.latest_experiment(client, ds.id, "candidate")
    reused = bool(base and cand)
    if not reused:
        print("No baseline/candidate experiments from challenge 5 found; running them now (a few minutes).")
        with ThreadPoolExecutor(max_workers=2) as pool:
            fb = pool.submit(run_one, client, "baseline", model, BASELINE_ID) if not base else None
            fc = pool.submit(run_one, client, "candidate", candidate_model, CANDIDATE_ID) if not cand else None
            base = fb.result() if fb else base
            cand = fc.result() if fc else cand
    return base, cand, reused


EDD_STAGE = {1: "Define", 2: "Define", 3: "Define", 4: "Measure", 5: "Change/Compare", 6: "Ship", 7: "Feed back"}

FRAMEWORK = [
    (1, "Frame the change", "opening slides; the scenarios in challenge 1"),
    (2, "Select representative traces", "challenges 1 and 2"),
    (3, "Build and version the golden dataset", "challenge 3"),
    (4, "Choose and trust the evaluators", "challenges 4, 6 and 7"),
    (5, "Run baseline vs candidate (quality, cost, latency)", "challenge 5"),
    (6, "Decide with a gate, and write down why", "challenge 8 (this notebook)"),
    (7, "Release safely and prove it after", "challenge 8 (post-release comparison)"),
]

OTHER_MIGRATIONS = [
    "Provider switch, e.g. Gemini to OpenAI (a mandated change): adapt prompts and parameters first, then steps 2-7 unchanged",
    "Version bump within a model family",
    "Larger to smaller model for cost: the cost line in the gate is the point",
    "Prompt or system-message change: the candidate is a prompt version",
    "Tool, retrieval or knowledge-base change: span-level evaluators catch tool-choice regressions",
    "Hosting or region change: quality should be unchanged; the gate checks latency, cost and parity",
    "Multi-model routing: run the experiment per route and gate each",
]


def render_report(report):
    """The one-page report as plain text (printed in the notebook; write it to a file if you want one)."""
    b, c = report["baseline"]["overall"], report["candidate"]["overall"]
    fmt = lambda v, f="{:.2f}": "-" if v is None else f.format(v)  # noqa: E731
    lines = [
        "MODEL UPGRADE REPORT",
        "=" * 60,
        f"1. Change      baseline {report['baseline_model']}  ->  candidate {report['candidate_model']}",
        f"2. Dataset     {report['dataset']} (test split, {b['n']} examples); experiments {report['baseline_experiment']} / {report['candidate_experiment']}",
        f"3. Evaluators  {EVALUATOR_VERSION}",
        "",
        "4. Results (overall)",
        f"   {'':22}{'baseline':>10}{'candidate':>11}",
        f"   {'quality':22}{fmt(b['quality']):>10}{fmt(c['quality']):>11}",
        f"   {'cost per example (USD)':22}{fmt(b['cost_per_example'], '{:.4f}'):>10}{fmt(c['cost_per_example'], '{:.4f}'):>11}",
        f"   {'quality per dollar':22}{fmt(b['quality_per_dollar'], '{:.0f}'):>10}{fmt(c['quality_per_dollar'], '{:.0f}'):>11}",
        f"   {'latency p95 (s)':22}{fmt(b['latency_p95'], '{:.1f}'):>10}{fmt(c['latency_p95'], '{:.1f}'):>11}",
        f"   {'error rate':22}{fmt(b['error_rate']):>10}{fmt(c['error_rate']):>11}",
        "   by segment (quality baseline -> candidate): " + ", ".join(
            f"{s} {fmt(report['baseline']['segments'][s]['quality'])}->{fmt(report['candidate']['segments'].get(s, {}).get('quality'))}"
            for s in report["baseline"]["segments"]),
        "   (cost uses the lab's assumed price table; replace with your rates)",
        "",
        f"5. DECISION    {report['outcome']}",
    ]
    lines += [f"   BLOCK      {r}" for r in report["blocks"]]
    lines += [f"   CONDITION  {r}" for r in report["conditions"]]
    lines += [
        "",
        "6. Rollout     stage it (small share or one segment), watch the same metrics, keep the baseline release ready to roll back",
        "7. Sign-offs   use case owner / model risk (record names and dates here)",
    ]
    return "\n".join(lines)


def run(decide=default_decide, quiet=False):
    client = Client()
    base_name, cand_name, reused = experiments(client)
    segs = metrics.example_segments(client, lab_sdk.ds_golden())
    base = metrics.summarize(metrics.experiment_rows(client, base_name, BASELINE_ID, segs))
    cand = metrics.summarize(metrics.experiment_rows(client, cand_name, CANDIDATE_ID, segs))
    outcome, blocks, conditions = _normalize(decide(base, cand))
    report = {
        "outcome": outcome, "pass": outcome != "BLOCKED", "blocks": list(blocks), "conditions": list(conditions),
        "reasons": list(blocks) + list(conditions),
        "baseline": base, "candidate": cand, "reused_experiments": reused,
        "baseline_experiment": base_name, "candidate_experiment": cand_name,
        "baseline_model": BASELINE_ID, "candidate_model": CANDIDATE_ID, "dataset": lab_sdk.ds_golden(),
        "thresholds": THRESHOLDS,
    }
    os.makedirs(lab_sdk.EXPORT_DIR, exist_ok=True)
    with open(f"{lab_sdk.EXPORT_DIR}/gate_report.json", "w") as f:
        json.dump(report, f, indent=2)
    if not quiet:
        print(render_report(report))
    return report


if __name__ == "__main__":
    sys.exit(0 if run()["pass"] else 1)
