"""Centralized model initialization.

`model` is the baseline agent; `judge_model` is the LLM judge (same model as the baseline);
`candidate_model` is the migration candidate.

Pick a provider with LLM_PROVIDER in `.env`:

  bedrock            Claude on Amazon Bedrock (default). Needs AWS credentials.
  openai_compatible  Any server that speaks the OpenAI chat API: Ollama, LM Studio, vLLM,
                     llama.cpp, or a hosted open-source endpoint. Set LLM_BASE_URL (and
                     LLM_API_KEY if the server wants one).
  ollama             Shorthand for openai_compatible against a local Ollama (http://localhost:11434/v1).

The models must support tool calling: the agent calls tools.
"""
import pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parents[1]

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=str(_ROOT / ".env"), override=False)

PROVIDER = os.environ.get("LLM_PROVIDER", "bedrock").strip().lower()

_DEFAULTS = {
    "bedrock": ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "us.anthropic.claude-sonnet-4-6"),
    "openai_compatible": ("llama3.2:3b", "llama3.1:8b"),
}
_DEFAULTS["ollama"] = _DEFAULTS["openai_compatible"]
if PROVIDER not in _DEFAULTS:
    raise SystemExit(f"LLM_PROVIDER={PROVIDER!r} is not supported. Use bedrock, openai_compatible or ollama.")

BASELINE_ID = os.environ.get("MODEL_BASELINE") or _DEFAULTS[PROVIDER][0]
CANDIDATE_ID = os.environ.get("MODEL_CANDIDATE") or _DEFAULTS[PROVIDER][1]

# USD per 1M tokens, as "input,output". An ASSUMPTION used by the cost step (challenges 5 and 8).
# Bedrock defaults are approximate list prices. Open-source models have no list price, so the
# defaults below are PLACEHOLDERS: set MODEL_BASELINE_PRICE / MODEL_CANDIDATE_PRICE to your GPU or
# hosting cost per 1M tokens, or the cost comparison is only illustrative.
_PRICE_DEFAULTS = {
    "bedrock": ("1.00,5.00", "3.00,15.00"),
    "openai_compatible": ("0.05,0.15", "0.20,0.60"),
}
_PRICE_DEFAULTS["ollama"] = _PRICE_DEFAULTS["openai_compatible"]


def _price(var, default):
    inp, out = (os.environ.get(var) or default).split(",")
    return {"input": float(inp), "output": float(out)}


BASELINE_PRICE = _price("MODEL_BASELINE_PRICE", _PRICE_DEFAULTS[PROVIDER][0])
CANDIDATE_PRICE = _price("MODEL_CANDIDATE_PRICE", _PRICE_DEFAULTS[PROVIDER][1])


def make_model(model_id: str):
    if PROVIDER == "bedrock":
        from langchain_aws import ChatBedrockConverse

        kwargs = {"model_id": model_id, "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
                  "max_tokens": 2048}
        if "anthropic" in model_id:
            kwargs["provider"] = "anthropic"
        return ChatBedrockConverse(**kwargs)

    from langchain_openai import ChatOpenAI

    base_url = os.environ.get("LLM_BASE_URL") or (
        "http://localhost:11434/v1" if PROVIDER == "ollama" else None)
    if not base_url:
        raise SystemExit("LLM_PROVIDER=openai_compatible needs LLM_BASE_URL in .env "
                         "(for Ollama: http://localhost:11434/v1).")
    return ChatOpenAI(model=model_id, base_url=base_url,
                      api_key=os.environ.get("LLM_API_KEY") or "not-needed",
                      max_tokens=2048, timeout=300, max_retries=2)


model = make_model(BASELINE_ID)
judge_model = make_model(BASELINE_ID)
candidate_model = make_model(CANDIDATE_ID)
