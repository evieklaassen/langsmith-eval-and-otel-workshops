"""Model provider selection for the agent and the LLM judge.

Pick a provider with LLM_PROVIDER in `.env`:

  bedrock            Claude on Amazon Bedrock (default). Needs AWS credentials.
  openai_compatible  Any server that speaks the OpenAI chat API: Ollama, LM Studio, vLLM,
                     llama.cpp, or a hosted open-source endpoint. Set LLM_BASE_URL (and
                     LLM_API_KEY if the server wants one).
  ollama             Shorthand for openai_compatible against a local Ollama (http://localhost:11434/v1).

Set the model with LLM_MODEL (or BEDROCK_MODEL_HAIKU for Bedrock). The model must support tool calling.
"""
import os

BEDROCK_DEFAULT = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
OSS_DEFAULT = "llama3.2:3b"


def provider():
    p = os.environ.get("LLM_PROVIDER", "bedrock").strip().lower()
    if p not in ("bedrock", "openai_compatible", "ollama"):
        raise SystemExit(f"LLM_PROVIDER={p!r} is not supported. Use bedrock, openai_compatible or ollama.")
    return p


def model_id():
    if provider() == "bedrock":
        return os.environ.get("LLM_MODEL") or os.environ.get("BEDROCK_MODEL_HAIKU") or BEDROCK_DEFAULT
    return os.environ.get("LLM_MODEL") or OSS_DEFAULT


def make_chat_model(temperature=0, max_tokens=512):
    if provider() == "bedrock":
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(model=model_id(), region_name=os.environ.get("BEDROCK_REGION", "us-east-1"),
                                   temperature=temperature, max_tokens=max_tokens)
    from langchain_openai import ChatOpenAI

    base_url = os.environ.get("LLM_BASE_URL") or ("http://localhost:11434/v1" if provider() == "ollama" else None)
    if not base_url:
        raise SystemExit("LLM_PROVIDER=openai_compatible needs LLM_BASE_URL in .env "
                         "(for Ollama: http://localhost:11434/v1).")
    return ChatOpenAI(model=model_id(), base_url=base_url, api_key=os.environ.get("LLM_API_KEY") or "not-needed",
                      temperature=temperature, max_tokens=max_tokens, timeout=300, max_retries=2)
