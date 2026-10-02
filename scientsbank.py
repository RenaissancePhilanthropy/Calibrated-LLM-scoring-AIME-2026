"""SciEntsBank test_ua loader for this package.

``load_eval()`` returns ``(texts, labels, label_names)``: a flat list of
sample strings, gold integer labels (0..K-1), and the K class names in
label-index order.

Used by the direct and judge methods, so they see exactly the same gold
labels and class orderings. The channel method loads the dataset itself
(``channel/run.py``), because it needs the raw question/reference/answer
fields separately rather than one rendered string.
"""

from __future__ import annotations

from datasets import load_dataset  # type: ignore[import-untyped]

HF_DATASET = "nkazi/SciEntsBank"
HF_SPLIT = "test_ua"
# Pinned revision — the data the paper's numbers were computed from. Every
# stage that touches the dataset goes through load_split(), so this hash
# appears exactly once.
HF_REVISION = "abaadf77345c5d68b73b630131a8ae164a45f3ab"


def load_split():
    """The raw HuggingFace rows, at the revision the paper used."""
    return load_dataset(HF_DATASET, split=HF_SPLIT, revision=HF_REVISION)


def render_item(question: str, reference: str, student: str) -> str:
    """Render one item as the prose string direct and judge score against.

    ``direct/per_question_cf.py`` calls this with a null string in place
    of the student answer, so the wording is load-bearing beyond
    ``load_eval``: it is part of the scored prompt in two stages.
    """
    return (
        "The question the student was answering is as follows: "
        f'"{question}". The correct and complete reference answer '
        f'that we compared against is as follows: "{reference}". '
        f'The student\'s answer was: "{student}".'
    )


def load_eval() -> tuple[list[str], list[int], list[str]]:
    """Load SciEntsBank test_ua, collapsed to 2-way and rendered as prose.

    The five original classes collapse to correct (class 0) vs incorrect
    (everything else), guarded by a check on the dataset's label order.
    """
    ds = load_split()
    expected = ["correct", "contradictory", "partially_correct_incomplete",
                "irrelevant", "non_domain"]
    hf_names = list(ds.features["label"].names)
    if hf_names != expected:
        raise ValueError(
            f"SciEntsBank label order changed: expected {expected}, got "
            f"{hf_names}. The 2-way collapse below would silently mislabel."
        )
    texts = [
        render_item(r["question"], r["reference_answer"], r["student_answer"])
        for r in ds
    ]
    labels = [0 if int(r["label"]) == 0 else 1 for r in ds]
    return texts, labels, ["correct", "incorrect"]
