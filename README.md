# LangSmith workshops: run the notebooks yourself

Self-serve versions of two hands-on LangSmith workshops originally delivered live on Instruqt. Missed the session, or want to
redo it at your own pace? Everything runs on your laptop, against your own LangSmith workspace and your own AWS account for the models.

| Workshop | Folder | Notebooks | Time | What you learn |
|---|---|---|---|---|
| **LangSmith SDK deep dive: eval-driven development** | [`sdk-deep-dive/`](sdk-deep-dive/) | 9 | ~3.5 h | Use the Python SDK to run a model upgrade end to end: select and redact traces, build golden datasets, run evaluators down to the span, compare baseline vs candidate on quality, cost and latency, calibrate an LLM judge, and gate the migration with a one-page report. |
| **LangChain + OpenTelemetry: enriching trace metadata** | [`otel-enrichment/`](otel-enrichment/) | 7 | ~2.5 h | Send a LangChain agent's traces to LangSmith over OpenTelemetry and enrich them before export (resource attributes, a `SpanProcessor` with baggage, span attributes in tool code, geography tags), then curate a golden dataset with evaluators, an LLM judge and annotation queues. |

The two workshops are independent: pick either, or do both. They use **separate Python environments** (different `langsmith` versions), so set each up on its own.

## What you need

| Requirement | Details |
|---|---|
| **Python 3.12+** and [**uv**](https://docs.astral.sh/uv/getting-started/installation/) | `uv` creates the environment and installs everything from the lockfile. |
| **A LangSmith account and API key** | Key starts with `lsv2_`. Create one at [smith.langchain.com](https://smith.langchain.com) under **Settings > API Keys**. Use a **personal workspace** if you can (see [Troubleshooting](#troubleshooting)). US, EU, APAC and AWS US regions are all detected automatically. |
| **An AWS account with Amazon Bedrock access** | The agents and LLM judges call Claude through Bedrock in `us-east-1`. You must enable model access for **Claude Haiku 4.5** (both workshops) and **Claude Sonnet 4.6** (SDK deep dive only) in the Bedrock console, and have credentials that allow `bedrock:InvokeModel` on the `us.` inference profiles. |

Running the notebooks makes real LLM calls and writes real data to your LangSmith workspace (projects, datasets, experiments, feedback, annotation queues). The data is synthetic.
The Bedrock cost is small (tens of agent runs per notebook, mostly Haiku) but not zero. Each workshop README lists what gets created so you can clean it up.

## Quickstart

```bash
git clone https://github.com/evieklaassen/langsmith-eval-and-otel-workshops.git
cd langsmith-eval-and-otel-workshops/sdk-deep-dive      # or: otel-enrichment

uv sync                                  # create .venv and install dependencies
cp .env.example .env                     # then edit .env: add AWS credentials (or set AWS_PROFILE)
uv run jupyter lab                       # opens JupyterLab; start with notebooks/01_*.ipynb
```

Notes:

- **Run the notebooks in order.** Later notebooks read datasets, experiments and traces that earlier ones created in your LangSmith workspace.
- **Notebook 1 asks for your LangSmith API key** with hidden input, checks it against each LangSmith region, and writes it to your local `.env`. You never put the LangSmith key in a file yourself.
- **Run every cell**, top to bottom. Each notebook ends with links into the LangSmith UI showing the effect of what you just did.
- If you edit a function and an enrichment or result doesn't change, use **Kernel > Restart Kernel and Run All Cells**.

## How this differs from the live workshop

- There is no instructions panel, no **Check** button, and no per-sandbox AWS account. The one cell per notebook that learners wrote live is **already filled in with a reference solution**, headed `# Reference solution: try writing this yourself first.` Cover it and write your own first if you want the exercise.
- You bring your own AWS credentials for Bedrock (`.env` or `AWS_PROFILE`).
- Notebooks refer to "challenges"; challenge N is notebook N.
- Optional cells (remote evaluators, Align Evals, a second enrichment run) depend on your LangSmith plan. If you don't have the feature, the cell prints a message or fails harmlessly; nothing later depends on it.

## Repo layout

```
sdk-deep-dive/            # workshop 1
├── notebooks/            # 01 ... 09
├── eval_lab/             # the order-support agent, golden cases, evaluators, metrics and the migration gate
├── helpers/              # plumbing: credential prompt, SDK helpers, enrichment registries, UI links
├── utils/models.py       # Bedrock model setup
├── pyproject.toml, uv.lock, .env.example
otel-enrichment/          # workshop 2
├── notebooks/            # 01 ... 07
├── helpers/              # the bank support agent, OTel setup, enrichment reference, coverage report, curation helpers
└── pyproject.toml, uv.lock, .env.example
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Agent cell fails with `403 ... requires explicit agent creation for tracing` | Your LangSmith workspace blocks tracing until an agent is registered (common in some organization workspaces) | Use a key from a personal workspace, or any workspace that allows creating tracing projects. Notebook 1 detects this and tells you. |
| Notebook 1 says the key didn't authenticate in any region | Wrong or revoked key, or a self-hosted LangSmith | Re-create the key and re-run the cell. For self-hosted LangSmith, set `LANGSMITH_ENDPOINT` in `.env` yourself. |
| `AccessDeniedException` / "model not available" from Bedrock | Model access not enabled, wrong region, or IAM policy missing | Enable Claude Haiku 4.5 (and Sonnet 4.6 for the SDK workshop) in the Bedrock console for `us-east-1`, and allow `bedrock:InvokeModel` on the inference profiles. Wait a minute after enabling and re-run. |
| `NoCredentialsError` | No AWS credentials visible to the Jupyter process | Put keys in `.env`, or export `AWS_PROFILE` (and run `aws sso login`) in the shell **before** `uv run jupyter lab`. |
| A cell finds no runs, or counts are short, right after a previous cell | LangSmith ingestion is eventually consistent | Wait a few seconds (up to two minutes for OTel traces) and re-run the cell. |
| Your organization's AWS policy denies `us.` profiles | Region or inference-profile restrictions | Set `MODEL_BASELINE` / `MODEL_CANDIDATE` (SDK) or `BEDROCK_MODEL_HAIKU` (OTel) in `.env` to a model id you're allowed to use. Tool-calling-capable Claude models work best. |

More detail is in each workshop's README.
