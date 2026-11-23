"""One place that points the OpenTelemetry SDK at LangSmith.

LangChain and LangGraph emit spans through OpenTelemetry when LANGSMITH_OTEL_ENABLED=true.
The spans go to whichever global TracerProvider exists when LangChain is first used, so
`configure()` must run BEFORE the agent is built, and only once per Python process
(restart the notebook kernel to change it).

The enrichment hooks all sit on that provider, before the exporter:

  Resource attributes   -> constants for the whole process (deployment, identity)
  SpanProcessor.on_start -> values that change per request (channel, user, geography)
  span.set_attribute     -> values only the running code knows (risk decisions)

LangSmith stores any span attribute named `langsmith.metadata.<key>` as run metadata
`<key>`. `to_langsmith_attributes` does that mapping.
"""

import json
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

SERVICE_NAME = "bank-support-agent"
_state = {"provider": None}


def otlp_endpoint():
    base = (os.environ.get("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com").rstrip("/")
    return f"{base}/otel/v1/traces"


def to_langsmith_attributes(meta):
    """Map {key: value} to {'langsmith.metadata.<key>': value}.

    Span attribute values must be str, bool, int or float, so anything else is JSON-encoded.
    """
    attrs = {}
    for key, value in meta.items():
        if not isinstance(value, (str, bool, int, float)):
            value = json.dumps(value)
        attrs[f"langsmith.metadata.{key}"] = value
    return attrs


def _exporter_processor():
    """The processor that ships spans to LangSmith. Prefers LangSmith's own; falls back to plain OTLP."""
    try:
        from langsmith.integrations.otel import OtelSpanProcessor

        return OtelSpanProcessor()
    except Exception:  # noqa: BLE001 - older langsmith: use the documented OTLP endpoint directly
        exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint(),
            headers={"x-api-key": os.environ["LANGSMITH_API_KEY"],
                     "Langsmith-Project": os.environ["LANGSMITH_PROJECT"]},
        )
        return BatchSpanProcessor(exporter)


def check_endpoint():
    """Send an empty (valid) OTLP request to LangSmith and print the URL and HTTP status.

    200 means the exporter will work. 404 means the URL is wrong for this key's region or deployment;
    401/403 means the key is rejected. The key itself is never printed.
    """
    import requests

    url = otlp_endpoint()
    try:
        r = requests.post(url, data=b"", timeout=10, headers={
            "x-api-key": os.environ.get("LANGSMITH_API_KEY", ""),
            "Langsmith-Project": os.environ.get("LANGSMITH_PROJECT", ""),
            "Content-Type": "application/x-protobuf"})
        print(f"OTLP endpoint check: POST {url} -> HTTP {r.status_code}")
        if r.status_code == 404:
            print("  404: this URL does not exist for your workspace. Check LANGSMITH_ENDPOINT (region) and re-run the key cell.")
        return r.status_code
    except Exception as exc:  # noqa: BLE001 - never print the body or headers
        print(f"OTLP endpoint check: could not reach {url} ({type(exc).__name__})")
        return None


def configure(resource_attributes=None, processors=()):
    """Install the global TracerProvider. Call once, before building the agent.

    resource_attributes  dict added to the OTel Resource (process-wide constants)
    processors           your SpanProcessors; they run, in order, before the exporter
    """
    if _state["provider"] is not None:
        print("otel_setup.configure: already configured in this kernel; restart the kernel to change it")
        return _state["provider"]
    os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    resource = Resource.create({"service.name": SERVICE_NAME, **(resource_attributes or {})})
    provider = TracerProvider(resource=resource)
    for p in processors:
        provider.add_span_processor(p)
    provider.add_span_processor(_exporter_processor())
    trace.set_tracer_provider(provider)
    _state["provider"] = provider
    check_endpoint()
    return provider


def flush():
    if _state["provider"] is not None:
        _state["provider"].force_flush()
