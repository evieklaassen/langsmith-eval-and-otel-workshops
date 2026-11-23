"""Links into the learner's own LangSmith workspace.

Every notebook ends with at least one of these, so learners see what the code
did in the LangSmith UI. URL shapes (/o/<org>/projects/p/<id>, /datasets/<id>,
/annotation-queues/<id>) are the UI's own; if one ever 404s the printed hint says
where to click instead. Never prints an exception body: a credential failure can
echo the key back.
"""

import os

from langsmith import Client


def ui_host():
    # The UI host mirrors the API host for the region: dropping "api." is the
    # whole mapping. `or` rather than a get() default: the variable is always
    # PRESENT in some environments and is empty for a US-region key.
    api = os.environ.get("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"
    return api.replace("api.smith", "smith")


def _project(client=None):
    client = client or Client()
    return client.read_project(project_name=os.environ["LANGSMITH_PROJECT"])


def project_url(client=None):
    p = _project(client)
    return f"{ui_host()}/o/{p.tenant_id}/projects/p/{p.id}"


def _show(title, url, hint):
    print(f"{title}:")
    print(f"  {url}")
    if hint:
        print(f"  -> {hint}")


def project_link(hint=None):
    """Print the tracing project URL, with an optional 'what to look at' hint."""
    try:
        _show("Your traces in LangSmith", project_url(), hint)
    except Exception:  # noqa: BLE001
        print(f"Open {ui_host()} and choose project:", os.environ.get("LANGSMITH_PROJECT"))
        if hint:
            print(f"  -> {hint}")


def run_link(run, hint=None):
    """Print the URL of one trace."""
    try:
        base = project_url()
        _show("This trace in LangSmith", f"{base}/r/{run.id}", hint)
    except Exception:  # noqa: BLE001
        project_link(hint)


def dataset_link(name, hint=None):
    """Print the URL of a dataset by name."""
    try:
        client = Client()
        ds = client.read_dataset(dataset_name=name)
        org = _project(client).tenant_id
        _show(f"Dataset '{name}' in LangSmith", f"{ui_host()}/o/{org}/datasets/{ds.id}", hint)
    except Exception:  # noqa: BLE001
        print(f"Open {ui_host()} > Datasets & Experiments and choose '{name}'.")
        if hint:
            print(f"  -> {hint}")


def queue_link(name, hint=None):
    """Print the URL of an annotation queue by name."""
    try:
        client = Client()
        q = next(iter(client.list_annotation_queues(name=name)))
        org = _project(client).tenant_id
        _show(f"Annotation queue '{name}' in LangSmith", f"{ui_host()}/o/{org}/annotation-queues/{q.id}", hint)
    except Exception:  # noqa: BLE001
        print(f"Open {ui_host()} > Annotation Queues and choose '{name}'.")
        if hint:
            print(f"  -> {hint}")
