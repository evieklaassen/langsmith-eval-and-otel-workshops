"""Centralized model initialization (Amazon Bedrock).

`model` is the baseline agent and the judge; `candidate_model` is the migration
candidate. Both run on Amazon Bedrock with your AWS credentials. Both models use US geo inference profiles: Bedrock model access is region-scoped (no global.* profiles needed).
"""
import pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parents[1]

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=str(_ROOT / ".env"), override=False)

from langchain_aws import ChatBedrockConverse

_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
BASELINE_ID = os.environ.get("MODEL_BASELINE", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
CANDIDATE_ID = os.environ.get("MODEL_CANDIDATE", "us.anthropic.claude-sonnet-4-6")


def make_model(model_id: str) -> ChatBedrockConverse:
    kwargs = {"model_id": model_id, "region_name": _REGION, "max_tokens": 2048}
    if "anthropic" in model_id:
        kwargs["provider"] = "anthropic"
    return ChatBedrockConverse(**kwargs)


model = make_model(BASELINE_ID)
judge_model = make_model(BASELINE_ID)
candidate_model = make_model(CANDIDATE_ID)
