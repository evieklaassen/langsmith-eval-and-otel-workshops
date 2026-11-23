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
| **A model provider: pick one** | **Option A, open-source models (no cloud account):** install [Ollama](https://ollama.com/download) and pull a tool-calling model. See [Using open-source models](#using-open-source-models-no-aws-needed). **Option B, Claude on Amazon Bedrock:** enable model access for **Claude Haiku 4.5** (both workshops) and **Claude Sonnet 4.6** (SDK deep dive only) in the Bedrock console for `us-east-1`, and use credentials that allow `bedrock:InvokeModel` on the `us.` inference profiles. |

Running the notebooks makes real LLM calls and writes real data to your LangSmith workspace (projects, datasets, experiments, feedback, annotation queues). The data is synthetic.
With Ollama the model cost is zero (it runs on your machine). With Bedrock it is small (tens of agent runs per notebook, mostly Haiku) but not zero. Each workshop README lists what gets created so you can clean it up.

## Quickstart

```bash
git clone https://github.com/evieklaassen/langsmith-eval-and-otel-workshops.git
cd langsmith-eval-and-otel-workshops/sdk-deep-dive      # or: otel-enrichment

uv sync                                  # create .venv and install dependencies
cp .env.example .env                     # then edit .env: choose a provider (Ollama or Bedrock, see below)
uv run jupyter lab                       # opens JupyterLab; start with notebooks/01_*.ipynb
```

Notes:

- **Run the notebooks in order.** Later notebooks read datasets, experiments and traces that earlier ones created in your LangSmith workspace.
- **Notebook 1 asks for your LangSmith API key** with hidden input, checks it against each LangSmith region, and writes it to your local `.env`. You never put the LangSmith key in a file yourself.
- **Run every cell**, top to bottom. Each notebook ends with links into the LangSmith UI showing the effect of what you just did.
- If you edit a function and an enrichment or result doesn't change, use **Kernel > Restart Kernel and Run All Cells**.

## Using open-source models (no AWS needed)

Both workshops default to Claude on Bedrock, but they also run on open-source models served locally by [Ollama](https://ollama.com), or by any server that speaks the OpenAI chat API (LM Studio, vLLM, llama.cpp, a hosted open-source endpoint). You choose in `.env`; the notebooks are identical.

**Ollama, step by step**

1. Install Ollama and make sure it's running (`ollama list` should work).
2. Pull the models. The SDK workshop compares two (a smaller baseline and a larger candidate); the OTel workshop uses one:

   ```bash
   ollama pull llama3.2:3b      # both workshops (SDK baseline; OTel agent and judge)
   ollama pull llama3.1:8b      # SDK workshop only (the migration candidate)
   ```

3. In `.env`, set the provider and models (commented examples are in `.env.example`):

   ```bash
   # sdk-deep-dive/.env
   LLM_PROVIDER=ollama
   MODEL_BASELINE=llama3.2:3b
   MODEL_CANDIDATE=llama3.1:8b

   # otel-enrichment/.env
   LLM_PROVIDER=ollama
   LLM_MODEL=llama3.2:3b
   ```

   You can leave the AWS lines commented out; they're ignored. Use `LLM_PROVIDER=openai_compatible` with `LLM_BASE_URL` (and `LLM_API_KEY` if the server needs one) for anything other than a default local Ollama. These dedicated `LLM_*` variables are used on purpose so a stray `OPENAI_BASE_URL` or `OPENAI_API_KEY` in your shell can never redirect or leak into the workshop.
4. Restart Jupyter after changing `.env`.

**What to expect with open-source models**

- **Tool calling is required.** The agents call tools, so pick a model that supports it (Llama 3.1/3.2, Qwen 2.5 and Mistral families do on Ollama). Models without tool support fail on the first agent cell.
- **Verified:** the agent, token accounting and judge calls were tested against Ollama with `llama3.2:3b`. The full notebooks were not run end to end with open-source models, and `llama3.1:8b` as the SDK candidate is a reasonable but untested default.
- **Results will differ from the Claude run.** Small models call tools less reliably and produce less consistent judge JSON, so scores, the judge-calibration numbers (SDK notebook 7) and the gate outcome (SDK notebook 8) can look worse or noisier. That is itself a useful lesson in why you measure. For a sturdier judge and agent, use a larger model (for example `llama3.1:8b` or bigger) as the baseline, and expect slower runs: local models are bounded by your hardware.
- **Cost step.** Open-source models have no list price, so the cost comparison (SDK notebooks 5 and 8) uses **placeholder prices** (`MODEL_BASELINE_PRICE`, `MODEL_CANDIDATE_PRICE`, USD per 1M tokens). Set them to your own GPU or hosting cost, or treat the cost numbers as illustrative.
- **Speed.** The notebooks run a few requests in parallel. Ollama serializes by default; set `OLLAMA_NUM_PARALLEL=2` (or more, if you have memory) before `ollama serve` to speed things up.

## How this differs from the live workshop

- There is no instructions panel, no **Check** button, and no per-sandbox AWS account. The one cell per notebook that learners wrote live is **already filled in with a reference solution**, headed `# Reference solution: try writing this yourself first.` Cover it and write your own first if you want the exercise.
- You bring your own model provider: open-source models via Ollama (no account), or your own AWS credentials for Bedrock (`.env` or `AWS_PROFILE`).
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
| Connection refused / `APIConnectionError` with Ollama | Ollama isn't running, or `LLM_BASE_URL` is wrong | Start Ollama (`ollama serve` or the desktop app); check `curl localhost:11434/v1/models`. |
| `model ... does not exist` (404) with Ollama | The model isn't pulled | `ollama pull <model>` for each model named in `.env` |
| Agent never calls tools, or errors on tools, with an open-source model | The model doesn't support tool calling | Choose a tool-capable model (see above) |
| `NoCredentialsError` | No AWS credentials visible to the Jupyter process | Put keys in `.env`, or export `AWS_PROFILE` (and run `aws sso login`) in the shell **before** `uv run jupyter lab`. |
| A cell finds no runs, or counts are short, right after a previous cell | LangSmith ingestion is eventually consistent | Wait a few seconds (up to two minutes for OTel traces) and re-run the cell. |
| Your organization's AWS policy denies `us.` profiles | Region or inference-profile restrictions | Set `MODEL_BASELINE` / `MODEL_CANDIDATE` (SDK) or `BEDROCK_MODEL_HAIKU` (OTel) in `.env` to a model id you're allowed to use, or switch to open-source models (above). |

More detail is in each workshop's README.
