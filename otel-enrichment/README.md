# Workshop 2: LangChain + OpenTelemetry, enriching trace metadata before it reaches LangSmith

**Time:** about 2.5 hours for all seven notebooks. **Level:** intermediate. **Model:** by default Claude Haiku 4.5 on Amazon Bedrock `us-east-1` (the agent and the LLM judge). An open-source model via Ollama works too (`llama3.2:3b` tested): see [Using open-source models](../README.md#using-open-source-models-no-aws-needed).

## The idea

A provided **LangChain bank support agent** (a knowledge-base retriever and an account tool) sends traces to your LangSmith workspace with the **OpenTelemetry SDK**
(`LANGSMITH_OTEL_ENABLED=true`, a global `TracerProvider`). OTel traces carry limited metadata out of the box. Notebook 1 measures what arrives against a list of **83 desired attributes**
(`helpers/phase2_requirements.csv`), then the rest of the workshop adds what's missing **before the spans leave the process**, in three places:

- **Resource attributes** for process-wide constants: deployment and agent identity (notebook 2).
- **A custom `SpanProcessor` with baggage** for per-request values: channel, user, conversation, caller (notebook 3).
- **Span attributes set in the agent's own code** for what only the running code knows: data classification, PII, guardrail, policy, human-in-the-loop, tool and retrieval details (notebook 4).

Notebook 5 adds customer geography through the same processor and builds traffic dashboards. Notebook 6 curates a golden dataset: a code evaluator scores traces,
an LLM judge grades the best candidates, and top scorers are added on a rolling window without duplicates. Notebook 7 sends uncertain traces to an annotation queue,
where reviewers add approved ones to the same dataset, then validates attribute coverage.

This workshop has no after-the-fact `update_run` enrichment; that approach is covered in the SDK deep dive (notebook 9). The agent data and prompts are synthetic.

## Setup

See the [top-level README](../README.md#what-you-need) for prerequisites. Then, from this folder:

```bash
uv sync
cp .env.example .env        # choose a provider: Ollama (no account) or Bedrock (AWS credentials / AWS_PROFILE)
uv run jupyter lab
```

Open `notebooks/01_setup_and_first_trace.ipynb`. The first cells ask for your LangSmith API key and a short name or initials (used to name your tracing project).

## Notebooks

| # | Notebook | You will |
|---|---|---|
| 1 | `01_setup_and_first_trace` | Run the agent with OTel tracing on, then produce the gap report against the 83 attributes |
| 2 | `02_resource_attributes` | Attach environment, cluster, pod and agent identity to every span with an OTel `Resource` |
| 3 | `03_span_processor_and_baggage` | Carry channel, user and conversation in OTel baggage and stamp them onto spans with a `SpanProcessor` |
| 4 | `04_span_attributes_in_tool_code` | Record data classification, PII, guardrail, policy and approval decisions from inside the agent |
| 5 | `05_geography_tags_and_dashboards` | Tag traces with customer country and region before export, then build Monitor charts grouped by region |
| 6 | `06_golden_datasets_with_evaluators` | Score traces with a code evaluator and an LLM judge; add the best to a dataset on a rolling window |
| 7 | `07_annotation_queues_to_datasets` | Send uncertain traces to an annotation queue, promote approved ones to the dataset, validate coverage (aim for 90%+ of the hands-on attributes) |

Run them in order. Each notebook has one function or class to write; a reference solution is already filled in, and later notebooks import the reference implementations from `helpers/enrichment_ref.py`,
so you can't get stuck on a notebook you skipped or got wrong. Notebook 5's dashboards are built in the LangSmith UI: the notebook links you to the right place.

## What gets created

In **your** LangSmith workspace: a tracing project (`otel-metadata-workshop-<your-name>-local`), a golden dataset, an annotation queue, and feedback rows. Dataset and queue names include `WORKSHOP_SFX` from `.env` (default `local`); change it to something unique if you share a workspace. Locally: your `.env`. Delete the project, dataset and queue from the LangSmith UI when you're done.

## Troubleshooting (workshop-specific)

| Symptom | Cause | Fix |
|---|---|---|
| No traces appear (notebook 1) | Key and project in different LangSmith regions, or ingestion lag | Re-run the key cell; wait up to two minutes; re-run the agent cell |
| `OTLP endpoint check ... 404` | The OTLP URL doesn't exist for your workspace region | Check `LANGSMITH_ENDPOINT` in `.env` and re-run the key cell |
| Enrichment keys missing after editing a function (notebooks 2-5) | `otel_setup.configure()` runs once per kernel, so an edited function isn't picked up | **Kernel > Restart Kernel and Run All Cells** |
| Only the root span carries per-request keys (notebook 3) | LangChain's spans aren't nested under the agent's root span in your SDK version | Fine: later notebooks only need the keys on runs |
| The LLM judge reports the model is unavailable (notebook 6) | Bedrock model access not enabled, region/profile not allowed, or Ollama not running / model not pulled | Enable Claude Haiku 4.5 in the Bedrock console for `us-east-1`, or set `BEDROCK_MODEL_HAIKU`; with Ollama, check `LLM_MODEL` and `ollama list` |
| Judge scores are all 0 with "unparseable judge output" (notebook 6) | A small open-source model isn't returning the JSON the judge asks for | Use a larger model via `LLM_MODEL`, then re-run the judge cell |
