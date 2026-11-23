"""Post-hoc enrichment: add metadata and tags to runs that are already in LangSmith.

The seeded production traces carry env, segment, release and thread_id, and nothing about who the
customer is or how risky the request was. That kind of fact often lives somewhere else (an order
system, a deployment record) or arrives after the trace was written, so it is added afterwards
with `client.update_run`.

The registries below stand in for your real systems of record. Swap them for a lookup against
your own.
"""

import re
import time

ENRICH_VERSION = "1"          # stamped on every enriched run, so a rerun skips what is done
ORDER_ID = re.compile(r"\bA\d{4}\b")

# Stand-in for an order system: who the customer behind each order is.
ORDER_REGISTRY = {
    "A1001": {"customer_country": "SG", "customer_tier": "gold"},
    "A1002": {"customer_country": "HK", "customer_tier": "standard"},
    "A1003": {"customer_country": "IN", "customer_tier": "standard"},
    "A1004": {"customer_country": "SG", "customer_tier": "standard"},
    "A1005": {"customer_country": "AU", "customer_tier": "gold"},
}
COUNTRY_REGION = {"SG": "sea", "HK": "north-asia", "IN": "south-asia", "AU": "anz"}

# Stand-in for a deployment record: what each release actually shipped.
DEPLOYMENTS = {
    "v1.0": {"prompt_version": "SYSTEM_PROMPT_V1", "date": "2026-09-01"},
    "v1.1": {"prompt_version": "SYSTEM_PROMPT_V2", "date": "2026-10-01"},
}


def meta_of(run):
    return dict((getattr(run, "extra", None) or {}).get("metadata") or {})


def is_enriched(run):
    return meta_of(run).get("enrichment_version") == ENRICH_VERSION


def query_of(run):
    return str((run.inputs or {}).get("query", ""))


def risk_class(run):
    """What the run did, not what the customer asked: money moved, or information only."""
    if run.error:
        return "unknown"
    trajectory = (run.outputs or {}).get("trajectory", [])
    return "financial" if "issue_refund" in trajectory else "informational"


def registry_metadata(run):
    """(metadata, tags) to add to a root run, from the registries and from the run itself."""
    found = ORDER_ID.search(query_of(run))
    customer = ORDER_REGISTRY.get(found.group(0), {}) if found else {}
    country = customer.get("customer_country", "unknown")
    region = COUNTRY_REGION.get(country, "unknown")
    release = meta_of(run).get("release")
    metadata = {
        "customer_country": country,
        "customer_region": region,
        "customer_tier": customer.get("customer_tier", "unknown"),
        "risk_class": risk_class(run),
        **{f"deploy_{k}": v for k, v in DEPLOYMENTS.get(release, {}).items()},
        "enrichment_version": ENRICH_VERSION,
    }
    return metadata, [f"region:{region}", f"risk:{metadata['risk_class']}"]


def apply(client, project, merge_into, limit=None):
    """Enrich every root run in `project` that is not enriched yet, using your `merge_into`.

    Returns (updated, already_done). Safe to run again: stamped runs are skipped.
    """
    runs = list(client.list_runs(project_name=project, is_root=True, limit=limit))
    updated = already = 0
    for run in runs:
        if is_enriched(run):
            already += 1
            continue
        metadata, tags = registry_metadata(run)
        for attempt in range(3):
            try:
                merge_into(run, metadata, tags)
                updated += 1
                break
            except Exception as exc:  # noqa: BLE001 - a run can lag behind ingestion
                if attempt == 2:
                    print(f"  could not update {run.name}: {type(exc).__name__}")
                time.sleep(2)
    flush = getattr(client, "flush", None)
    if callable(flush):
        flush()
    return updated, already


def wait_until_enriched(client, project, expected, timeout=90):
    """Poll until `expected` root runs show the stamp (updates are applied asynchronously)."""
    end = time.time() + timeout
    seen = []
    while time.time() < end:
        seen = [r for r in client.list_runs(project_name=project, is_root=True) if is_enriched(r)]
        if len(seen) >= expected:
            break
        time.sleep(3)
    return seen
