"""Pipeline B — brain-aligned embeddings from TRIBE v2.

Two layers, deliberately separated so the analysis does not depend on torch:

* Pure-math (numpy/scipy/sklearn): ``metrics``, ``stats``, ``pooling``. These
  operate on already-extracted embedding matrices and are fully unit-tested.
* Model-dependent (torch + ``tribev2``): ``extraction``, ``baselines``,
  ``validation``. These tap a live checkpoint. Imported lazily so that
  ``import pipeline_b`` works without torch installed.

The ``experiments`` module wires E0–E5 from the design doc over embedding
matrices; it needs only the pure-math layer.
"""

from . import metrics, pooling, stats, experiments, contamination

__all__ = ["metrics", "pooling", "stats", "experiments", "contamination"]

# Model-dependent submodules (extraction, baselines, validation) are NOT
# imported here so the package stays importable without torch. Import them
# explicitly, e.g. ``from pipeline_b import extraction``.
