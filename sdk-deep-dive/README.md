# Workshop 1: LangSmith SDK deep dive (eval-driven development)

**Time:** about 3.5 hours for all nine notebooks. **Level:** intermediate. **Models:** Claude Haiku 4.5 (baseline agent and judge) and Claude Sonnet 4.6 (migration candidate), on Amazon Bedrock `us-east-1`.

## The idea

Eval-driven development (EDD): define what good looks like, measure, change, compare, ship, and feed production failures back in. You apply it to one concrete job, **a model upgrade**,
using a repeatable seven-step framework so results are comparable across teams. It starts from one of two situations:

- **Proactive:** quality is degrading or a new requirement arrives. *Would a different model fix it, and at what cost?*
- **Reactive:** a mandated upgrade (for example, switching provider). *Is the mandated model safe to ship?*

| Step | What | Notebook |
|---|---|---|
| 1 | Frame the change (trigger, baseline, candidate, success criteria) | 1 (intro) |
| 2 | Select representative traces | 1, 2 |
| 3 | Build and version the golden dataset | 3 |
| 4 | Choose and trust the evaluators | 4 (build), 6 and 7 (trust) |
| 5 | Run baseline vs candidate on quality, **cost** and latency, by segment | 5 |
| 6 | Decide with a gate, and write down why | 8 |
| 7 | Release safely and prove it after | 8 |
| Extension | Enrich traces after the fact with segments production traces never carried | 9 |

You work with one **order-support agent** with four tools (`lookup_order`, `check_refund_eligibility`, `issue_refund`, `search_faq`) over in-memory data, so runs are fast and repeatable.
Two prompt versions ship as releases `v1.0` (skips the eligibility check) and `v1.1` (enforces it). Notebook 1 seeds about 26 traces across both releases, with tags, three multi-turn threads, a few errors,
and fake customer emails and phone numbers in the inputs. Only synthetic data is sent.

## Setup

See the [top-level README](../README.md#what-you-need) for prerequisites. Then, from this folder:

```bash
uv sync
cp .env.example .env        # add AWS credentials, or set AWS_PROFILE in your shell
uv run jupyter lab
```

Open `notebooks/01_setup_and_trace_selection.ipynb`. The first cells ask for your LangSmith API key and a short name or initials
(used to name your tracing project).

## Notebooks

| # | Notebook | You will | ~Time |
|---|---|---|---|
| 1 | `01_setup_and_trace_selection` | Connect LangSmith, seed traces, select runs with filters (tags, metadata, status, date) | 15 min |
| 2 | `02_full_run_tree_and_redaction` | Fetch a whole run tree, redact emails and phone numbers, export to JSON, JSONL and Parquet | 20 min |
| 3 | `03_datasets_versions_and_splits` | Build datasets from traces and from code, tag versions, use splits, pull and push | 20 min |
| 4 | `04_evaluators` | Run code and LLM-as-judge evaluators; score individual tool-call spans and whole threads | 25 min |
| 5 | `05_baseline_vs_candidate` | Run two models on the same golden split; compare quality, cost and latency by segment; pairwise comparison; upload outside results; prompt versions | 35 min |
| 6 | `06_feedback_and_annotation_queues` | Capture feedback, write the rules that pick runs worth a human's time, fill an annotation queue | 20 min |
| 7 | `07_judge_calibration` | Measure judge-vs-human agreement (accuracy and Cohen's kappa), adjudicate, retune the judge | 25 min |
| 8 | `08_gated_migration_and_report` | Write the gate (`PASS` / `PASS WITH CONDITIONS` / `BLOCKED`), print the one-page report, compare releases | 25 min |
| 9 | `09_enrich_after_the_fact` | Add geography, tier and risk class to existing traces with `update_run` (read, merge, write), idempotently | 15 min |

**Run them in order.** Notebooks 5 and 8 reuse the golden dataset from notebook 3 and the experiments from notebook 5. If you only have about two hours, notebooks 1, 3, 4, 5, 6 and 8 carry the core story,
but you still need to run the ones they build on; there is no shortcut that rebuilds missing state outside the live workshop.

The gate can also run outside the notebook: `uv run python -m eval_lab.gate` exits 0 for `PASS` and `PASS WITH CONDITIONS`, and 1 for `BLOCKED`.
Cost uses a **stated price table in `eval_lab/metrics.py` that is an assumption**. Replace it with your contract rates for a real decision.

## What gets created

In **your** LangSmith workspace: a production-style tracing project (`langsmith-sdk-workshop-<your-name>-local-prod`), datasets `sdk-from-traces-local`, `sdk-golden-local` and `sdk-uploaded-local`,
experiments (`eval-local`, `baseline`, `candidate`, plus pairwise and external-upload experiments), feedback rows, an annotation queue (`sdk-review-local`) and a prompt (`order-support-local`, if your plan allows prompts).
Locally: `exports/` (redacted trees, JSONL, Parquet, reports) and your `.env`.
**Sharing a workspace with others?** Dataset, queue and prompt names end in `WORKSHOP_SFX` from `.env` (default `local`). Change it to something unique before notebook 1 so you don't collide.
Delete the projects and datasets from the LangSmith UI when you're done.

## Optional cells and plan limits

Remote evaluators (notebook 4), Align Evals (notebook 7) and the second enrichment run (notebook 9) are marked **Optional**. Annotation queues and prompt versioning depend on your LangSmith plan.
To skip all optional cells, set `LAB_SKIP_OPTIONAL=1` in the shell before starting Jupyter.

## SDK version

The workshop pins `langsmith==0.14.11`. In that release `Client.list_runs()` and `Client.read_run()` are deprecated in favour of `client.runs.query()` and `client.runs.retrieve()`,
with removal scheduled after 2027-01-31. The notebooks use the deprecated calls and suppress the warnings. If you adapt this code, plan a port before that date.

## Troubleshooting (workshop-specific)

| Symptom | Cause | Fix |
|---|---|---|
| Notebook 5 comparison has no cost or latency figures | Runs have no token counts, or `run_cost()` wasn't run first | Run the `run_cost` cell, then re-run the "Compare by segment" cell |
| Notebook 7 agreement looks low | LLM judge variance | Re-run the judge v2 cell, or tighten the v2 prompt |
| Candidate tool calling fails | Sonnet 4.6 not enabled in your Bedrock account, or not tool-capable through Converse | Enable it, or set `MODEL_CANDIDATE` in `.env` to another tool-capable model |
| Warning about deprecated `list_runs` | Expected with the pinned SDK | Ignore; see [SDK version](#sdk-version) |
