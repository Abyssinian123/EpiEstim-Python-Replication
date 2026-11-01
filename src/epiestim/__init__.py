"""Core engine of the Nash et al. (2023) Python replication.

Modules:
- ``epiestim.py``: ``estimate_R`` (analytic Gamma posterior) and
  ``estimate_R_agg`` (the paper's EM reconstruction from temporally
  aggregated incidence), plus ``aggregate_inc``, ``discr_si`` and
  ``simulate_incidence``.
- ``loess.py``: line-by-line port of R's ``loess()`` (1-D, degrees 0/1/2).
- ``rds_reader.py``: minimal RDS reader used by the validation scripts.

The engine reproduces the paper's stored R results to 1e-14; see
``docs/replication_notes.md`` for the two serial-interval discretisations
(``si_variant``), the window-drop semantics and the simulation conventions.
"""

from .epiestim import (
    aggregate_inc,
    discr_si,
    estimate_R,
    estimate_R_agg,
    simulate_incidence,
    # internals re-exported because tests monkeypatch them:
    _get_r_from_R,  # noqa: F401
    _r2r0,          # noqa: F401
)
from . import loess, rds_reader  # noqa: F401

__all__ = [
    "estimate_R", "estimate_R_agg", "aggregate_inc", "discr_si",
    "simulate_incidence", "loess", "rds_reader",
]
