"""Training-set contamination control (§5.1) — the second critical threat.

TRIBE v2 trained on 500+ hours / 700+ subjects. The repo names four studies;
ANY evaluation on them measures train-set performance, not generalization.
Rule of the doc: when in doubt, treat a dataset as contaminated. The cost of a
false positive is a wasted project.

This module is a guardrail, not an oracle: it hard-blocks the four known
studies and forces an explicit, logged acknowledgement for anything else.
"""

from __future__ import annotations

# The four studies named in the repo's studies/ directory (§5.1). Known-contaminated.
KNOWN_TRAINING_STUDIES = frozenset(
    {"algonauts2025", "wen2017", "lahner2024", "lebel2023"}
)


class ContaminationError(RuntimeError):
    """Raised when an evaluation would run on known training data."""


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def is_known_contaminated(dataset_name: str) -> bool:
    key = _norm(dataset_name)
    return any(_norm(s) in key or key in _norm(s) for s in KNOWN_TRAINING_STUDIES)


def assert_clean(dataset_name: str, audited_ok: bool = False) -> None:
    """Gate an evaluation dataset before use.

    * Known training studies -> always blocked.
    * Anything else -> blocked unless the caller passes ``audited_ok=True``,
      which asserts a human checked the v2 corpus list (§5.1 mitigation 2) and
      confirmed this dataset is absent. Forces the audit to be a deliberate act.
    """
    if is_known_contaminated(dataset_name):
        raise ContaminationError(
            f"{dataset_name!r} is a known TRIBE v2 training study "
            f"({sorted(KNOWN_TRAINING_STUDIES)}). Any result here is train-set "
            "performance — pick an uncontaminated dataset (§5.1)."
        )
    if not audited_ok:
        raise ContaminationError(
            f"{dataset_name!r} is not on the known-training list, but the v2 "
            "corpus has not been audited for it. Confirm it is absent from the "
            "full training corpus, then pass audited_ok=True (§5.1). When in "
            "doubt, treat as contaminated."
        )
