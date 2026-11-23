"""Links into the learner's own LangSmith workspace (project, dataset, experiments, annotation queue).

Printed at the end of notebooks so learners can see the effect of the code in the UI.
URL shapes (/o/<org>/projects/p/<id>, /datasets/<id>, /annotation-queues/<id>) are the UI's own;
if one ever 404s, the printed hint says where to click instead. Never prints an exception body.
"""

import os

from langsmith import Client


def ui_host():
    api = os.environ.get("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"
    return api.replace("api.smith", "smith")


def _show(title, url, hint):
    print(f"{title}:\n  {url}")
    if hint:
        print(f"  -> {hint}")


def project_link(project_name, hint=None):
    try:
        p = Client().read_project(project_name=project_name)
        _show(f"Project '{project_name}' in LangSmith", f"{ui_host()}/o/{p.tenant_id}/projects/p/{p.id}", hint)
    except Exception:  # noqa: BLE001
        print(f"Open {ui_host()} > Tracing and choose project '{project_name}'.")
        if hint:
            print(f"  -> {hint}")


def dataset_link(dataset_name, hint=None):
    try:
        c = Client()
        ds = c.read_dataset(dataset_name=dataset_name)
        _show(f"Dataset '{dataset_name}' in LangSmith", f"{ui_host()}/o/{ds.tenant_id}/datasets/{ds.id}", hint)
    except Exception:  # noqa: BLE001
        print(f"Open {ui_host()} > Datasets & Experiments and choose '{dataset_name}'.")
        if hint:
            print(f"  -> {hint}")


def queue_link(queue_name, hint=None):
    try:
        c = Client()
        q = next(iter(c.list_annotation_queues(name=queue_name)))
        org = c.read_project(project_name=os.environ["LANGSMITH_PROJECT"] + "-prod").tenant_id
        _show(f"Annotation queue '{queue_name}' in LangSmith", f"{ui_host()}/o/{org}/annotation-queues/{q.id}", hint)
    except Exception:  # noqa: BLE001
        print(f"Open {ui_host()} > Annotation Queues and choose '{queue_name}'.")
        if hint:
            print(f"  -> {hint}")
