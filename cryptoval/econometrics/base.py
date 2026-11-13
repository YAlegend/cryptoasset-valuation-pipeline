"""The pair contract (spec 11.4), with no estimator in it.

A PAIR is the unit of this plan: a dependent series, an independent series, and the
coefficient the practitioner ratio forces on their relationship. Engle-Granger ESTIMATES
that coefficient; the ratio test assumes it. Holding both in one object is what lets the
chapter report them as the unconstrained and constrained versions of one hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class Unusable(Exception):
    """The pair cannot be tested on this asset, and the message carries the measurement.

    Deliberately the same shape as scoring.base.Unfittable: the runner turns it into a
    refusal ROW rather than letting it end the run, so a refusal is published with its
    cause instead of disappearing.
    """


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    dependent: str
    independent: str
    # The coefficient the practitioner RATIO forces: 1 for market cap over realized cap,
    # 2 for Metcalfe, whose law states value is proportional to the square of the users.
    forced_coefficient: float
    # The value the law predicts, where a law predicts one. Metcalfe's 2 is testable
    # against the estimated coefficient's interval; MVRV predicts no particular number.
    expected_coefficient: float | None = None
    citation: str = ""
    params: dict[str, Any] = field(default_factory=dict)
