"""Golden cases for the order-support agent. One source of truth: the notebooks,
the gate, and the notebooks all import from here."""

def _ex(query, trajectory, keywords, segment, split):
    return {"inputs": {"query": query}, "outputs": {"trajectory": trajectory, "keywords": keywords},
            "metadata": {"segment": segment}, "split": split}


_REFUND = ["lookup_order", "check_refund_eligibility", "issue_refund"]
_NOT_ELIGIBLE = ["lookup_order", "check_refund_eligibility"]

# 15 cases: 5 train, 10 test. Challenge 3 adds 2 more test cases, so the test split the
# experiments and the gate use has 12 (4 per segment: enough to compare by segment).
EXAMPLES = [
    # train
    _ex("What's the status of order A1001?", ["lookup_order"], ["delivered"], "order-status", "train"),
    _ex("Is order A1001 delivered yet?", ["lookup_order"], ["delivered"], "order-status", "train"),
    _ex("I'd like to return order A1001 for a refund.", _REFUND, ["refund"], "refund", "train"),
    _ex("What is the return window?", ["search_faq"], ["30 days"], "faq", "train"),
    _ex("Does the warranty cover accidental damage?", ["search_faq"], ["1-year"], "faq", "train"),
    # test
    _ex("Where is order A1003?", ["lookup_order"], ["shipped"], "order-status", "test"),
    _ex("Can you tell me the status of order A1002?", ["lookup_order"], ["delivered"], "order-status", "test"),
    _ex("What's the status of order A1004?", ["lookup_order"], ["delivered"], "order-status", "test"),
    _ex("I want a refund for order A1001. It arrived 5 days ago.", _REFUND, ["refund"], "refund", "test"),
    _ex("Please refund order A1002.", _NOT_ELIGIBLE, ["30 days"], "refund", "test"),
    _ex("Refund order A1003 please.", _NOT_ELIGIBLE, ["shipped"], "refund", "test"),
    _ex("What is your return policy?", ["search_faq"], ["30 days"], "faq", "test"),
    _ex("How long does shipping take?", ["search_faq"], ["3-5"], "faq", "test"),
    _ex("What warranty comes with your products?", ["search_faq"], ["1-year"], "faq", "test"),
    _ex("Do you offer express shipping?", ["search_faq"], ["1-2"], "faq", "test"),
]

# Added in challenge 3 to create a second dataset version.
EXTRA_EXAMPLES = [
    _ex("Refund my USB-C cable, order A1004.", _REFUND, ["refund"], "refund", "test"),
    _ex("Track order A1004 and tell me its total.", ["lookup_order"], ["24"], "order-status", "test"),
]

# Outputs from an agent that runs OUTSIDE LangSmith, uploaded in the challenge 5.
EXTERNAL_RESULTS = [
    {"answer": "Order A1003 has shipped.", "trajectory": ["lookup_order"], "score": 1},
    {"answer": "Order A1001 was delivered.", "trajectory": ["lookup_order"], "score": 1},
    {"answer": "Refund issued.", "trajectory": ["issue_refund"], "score": 0},
    {"answer": None, "trajectory": [], "score": 0, "error": "TimeoutError: upstream agent timed out"},
]
