"""Family B -- network / adoption (spec 10.3).

beta is ESTIMATED, never assumed. Metcalfe's law implies 2.0; published estimates give
about 1.69, and the gap matters because the exponent is what the valuation turns on.

The honest caveat travels in the diagnostics rather than the prose: an active address is
not a user, and a high R-squared between two trending series is nearly guaranteed. So
every fit reports beta's confidence interval, R-squared, and the Durbin-Watson statistic,
with residuals_autocorrelated set when DW says the residuals are not independent -- which
is the condition under which the R-squared is not evidence of anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoval.methods.base import Estimate
from cryptoval.methods.benchmarks import fit_log_log
from cryptoval.methods.registry import load_method_specs, param_set_hash, register

_SPECS = load_method_specs()


def fit_metcalfe(market_cap: pd.Series, addresses: pd.Series) -> dict:
    """OLS of log(market_cap) on log(active_addresses).

    The regression itself is ``fit_log_log``; only the guards live here, because their
    messages become this method's refusal reasons and the shared helper returns None.
    """
    cap = pd.to_numeric(market_cap, errors="coerce")
    adr = pd.to_numeric(addresses, errors="coerce")
    usable = pd.DataFrame({"cap": cap, "adr": adr}).dropna()
    usable = usable[(usable["cap"] > 0) & (usable["adr"] > 0)]
    if len(usable) < 30:
        raise ValueError(f"only {len(usable)} usable observations; need at least 30")
    if usable["adr"].nunique() < 2:
        raise ValueError("active_addresses has no variation, so beta is not identified")

    fit = fit_log_log(usable["adr"], usable["cap"], min_rows=30)
    if fit is None:  # unreachable: the guards above are fit_log_log's own conditions
        raise ValueError(f"log-log fit failed on {len(usable)} usable observations")
    return fit


@register
class MetcalfeEstimated:
    spec = _SPECS["metcalfe_estimated"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        minimum = int(params.get("min_fit_rows", 180))
        refit_every = int(params.get("refit_every_rows", 30))
        phash = param_set_hash(params)
        out: list[Estimate] = []
        fit: dict | None = None
        last_error = "insufficient trailing history"

        for position, row in enumerate(frame.itertuples(index=False)):
            vintage = getattr(row, "panel_vintage_ts", None)
            # Expanding TRAILING window: rows 0..position-1 only, so no row is fitted on
            # its own future. Refitting every N rows keeps a 4,000-day backfill tractable
            # without ever reaching forward.
            if position >= minimum and (fit is None or position % refit_every == 0):
                history = frame.iloc[:position]
                try:
                    fit = fit_metcalfe(history["market_cap_usd"], history["active_addresses"])
                except ValueError as exc:
                    fit, last_error = None, str(exc)

            addresses = pd.to_numeric(pd.Series([row.active_addresses]), errors="coerce").iloc[0]
            supply = float(row.supply) if row.supply else None
            unusable = (
                fit is None
                or supply is None
                or supply <= 0
                or not np.isfinite(addresses)
                or addresses <= 0
            )
            if unusable:
                out.append(
                    Estimate(
                        symbol=row.symbol,
                        metric_date=row.metric_date,
                        method_id=self.spec.method_id,
                        param_set_hash=phash,
                        fair_value=None,
                        band_low=None,
                        band_high=None,
                        inputs_used={},
                        diagnostics={},
                        applicable=False,
                        reason=last_error if fit is None else "missing addresses or supply",
                        panel_vintage_ts=vintage,
                    )
                )
                continue

            log_adr = float(np.log(addresses))
            implied = float(np.exp(fit["alpha"] + fit["beta"] * log_adr))
            low = float(np.exp(fit["alpha"] + fit["beta_ci_low"] * log_adr))
            high = float(np.exp(fit["alpha"] + fit["beta_ci_high"] * log_adr))
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=implied / supply,
                    band_low=min(low, high) / supply,
                    band_high=max(low, high) / supply,
                    inputs_used={"active_addresses": float(addresses), "supply": supply},
                    diagnostics={**fit, "fit_rows": fit["n"]},
                    applicable=True,
                    panel_vintage_ts=vintage,
                )
            )
        return out
