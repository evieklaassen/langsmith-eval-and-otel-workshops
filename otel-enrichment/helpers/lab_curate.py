"""Golden-dataset and annotation-queue plumbing for challenges 6 and 7.

The scoring logic and the choice of what to add stay in the notebooks. These
helpers hide the SDK calls that are not the point of the exercise.
"""

import datetime as dt
import json
import os
import time

from langsmith import Client

from enrich import answer_of, documents_of, meta_of, question_of

SFX = os.environ.get("WORKSHOP_SFX", "lab")
DATASET_NAME = f"agent-golden-{SFX}"
QUEUE_NAME = f"agent-review-{SFX}"


def ensure_dataset(client=None, name=DATASET_NAME):
    client = client or Client()
    if client.has_dataset(dataset_name=name):
        return client.read_dataset(dataset_name=name)
    return client.create_dataset(
        dataset_name=name,
        description="Golden examples curated from bank support agent traces (workshop).",
    )


def source_run_ids(client, dataset):
    """Run ids already in the dataset, so a repeat run never adds a duplicate."""
    seen = set()
    for ex in client.list_examples(dataset_id=dataset.id):
        rid = (ex.metadata or {}).get("source_run_id")
        if rid:
            seen.add(str(rid))
    return seen


def run_to_example(run, answer_of, source, extra=None):
    """(inputs, outputs, metadata) for one run, tagged with where it came from."""
    meta = {"source_run_id": str(run.id), "source": source,
            "curated_on": dt.date.today().isoformat(), **(extra or {})}
    inputs = {"question": question_of(run)}
    return inputs, {"answer": answer_of(run)}, meta


def add_runs(client, dataset, runs, answer_of, source, extra_by_run=None):
    """Add runs as examples, skipping any already present. Returns the number added."""
    seen = source_run_ids(client, dataset)
    fresh = [r for r in runs if str(r.id) not in seen]
    if not fresh:
        return 0
    rows = [run_to_example(r, answer_of, source, (extra_by_run or {}).get(str(r.id))) for r in fresh]
    client.create_examples(
        dataset_id=dataset.id,
        examples=[{"inputs": i, "outputs": o, "metadata": m} for i, o, m in rows],
    )
    return len(fresh)


def ensure_queue(client=None, name=QUEUE_NAME):
    client = client or Client()
    for q in client.list_annotation_queues(name=name):
        return q
    return client.create_annotation_queue(
        name=name,
        description="Agent answers a reviewer should look at before they join the golden dataset.",
    )


def retrieval_scores(trace_runs):
    """Every retrieval score in the trace: from retriever runs' documents, or from a
    `retrieval_scores` metadata value on any run (list, or the JSON string OTel carries)."""
    scores = [d.get("metadata", {}).get("score")
              for r in trace_runs if r.run_type == "retriever" for d in documents_of(r)]
    for r in trace_runs:
        v = meta_of(r).get("retrieval_scores")
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                v = None
        scores += v if isinstance(v, list) else []
    return [x for x in scores if isinstance(x, (int, float))]


def reference_score(root, trace_runs):
    """The scoring rubric from challenge 6, packaged for the notebooks and later notebooks after it."""
    if root.error:
        return 0.0
    if meta_of(root).get("policy_decision", "allow") != "allow":
        return 0.0
    score = 0.5
    scores = retrieval_scores(trace_runs)
    if scores:
        score += 0.3 * sum(scores) / len(scores)
    if not any(r.error for r in trace_runs if r.run_type == "tool" or (r.name or "").endswith(".audit")):
        score += 0.2
    return round(score, 3)


JUDGE_PROMPT = """You are grading a customer-support assistant's answer for a bank.

Question: {question}

Retrieved context:
{context}

Answer: {answer}

Score the answer from 1 to 5:
5 = correct, complete, grounded in the context, appropriate tone
3 = partly correct or not clearly grounded
1 = wrong, unsafe, or unhelpful

Reply with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def judge_prompt(run, trace_runs):
    context = "\n".join(d.get("page_content", "") for r in trace_runs if r.run_type == "retriever" or (r.name or "").startswith("search_kb")
                        for d in documents_of(r)) or "(none)"
    return JUDGE_PROMPT.format(question=question_of(run), context=context, answer=answer_of(run))


def ask_judge(prompt, retries=3):
    """One call to the judge model on Amazon Bedrock. Returns the reply text.

    Temperature 0 so the same answer gets the same score. Raises after `retries`.
    """
    import boto3

    client = boto3.client("bedrock-runtime", region_name=os.environ.get("BEDROCK_REGION", "us-east-1"))
    last = None
    for attempt in range(retries):
        try:
            resp = client.converse(
                modelId=os.environ.get("BEDROCK_MODEL_HAIKU", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 200, "temperature": 0},
            )
            return resp["output"]["message"]["content"][0]["text"]
        except Exception as exc:  # noqa: BLE001 - never print the body
            last = type(exc).__name__
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"judge model unavailable ({last})")


def parse_judge(text):
    """(score 1-5, reason) from the judge's JSON reply; (0.0, note) if it is not parseable."""
    try:
        data = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return float(data["score"]), str(data["reason"])
    except Exception:  # noqa: BLE001
        return 0.0, "unparseable judge output"


def reference_judge(question_run, trace_runs):
    """The judge from challenge 6, packaged for the notebook after it and later notebooks."""
    try:
        return parse_judge(ask_judge(judge_prompt(question_run, trace_runs)))
    except Exception as exc:  # noqa: BLE001
        return 0.0, str(exc)
