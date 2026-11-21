"""Cross-sectional, sector-cohorted scores (spec 10.5).

Every other method in this repo is a function of one asset's own history, which is what
run_methods' groupby("symbol") encodes. A cohort z-score is not: it needs every symbol on
one date, and its answer for one asset moves when another asset's data is revised.

That is the only thing special about it. Same estimate() signature, same Estimate rows --
the registry just hands it the whole panel instead of one symbol's slice.
"""

from __future__ import annotations

import pandas as pd

from cryptoval.methods.base import Estimate
from cryptoval.methods.multiples import _annualised
from cryptoval.methods.registry import load_method_specs, param_set_hash, register

_SPECS = load_method_specs()

RATIOS = {
    # numerator / annualised trailing denominator, both columns of the widened panel.
    "market_cap_to_revenue": ("market_cap_usd", "net_revenue_usd"),
    "market_cap_to_fees": ("market_cap_usd", "total_fees_usd"),
}


@register
class CohortZScore:
    spec = _SPECS["sector_cohort_zscore"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        minimum = int(params.get("min_cohort_size", 3))
        window = int(params.get("window_rows", 365))
        numerator_col, denominator_col = RATIOS[params.get("ratio", "market_cap_to_revenue")]
        phash = param_set_hash(params)

        work = frame.sort_values(["symbol", "metric_date"]).copy()
        # A TRAILING ANNUALISED denominator, per symbol, exactly as every other multiple in
        # Family D uses. One day's revenue would move an asset's own z-score by 0.26 sigma a
        # day on average and more than a full sigma on 5% of days (measured on the defi
        # cohort, 2026-10-16) -- that is yesterday's fee receipts talking, not relative
        # valuation. The trailing version moves 0.022. Annualising also makes the ratio read
        # on the same scale as price_to_sales rather than 365x away from it.
        denominator = work.groupby("symbol", sort=False)[denominator_col].transform(
            lambda series: _annualised(series, window)
        )
        # Not a column on `work`: itertuples renames any leading-underscore column to a
        # positional name, and a non-underscored one could collide with a panel column.
        work = work.assign(
            cohort_ratio=pd.to_numeric(work[numerator_col], errors="coerce")
            / denominator.where(denominator > 0)
        )
        ratios = work["cohort_ratio"].to_numpy()

        # Cohort statistics per (date, sector) over the members with a usable ratio.
        # ddof=0: the cohort IS the population here, not a sample drawn from a larger one.
        usable = work.dropna(subset=["cohort_ratio"])
        stats = {
            key: (len(group), float(group.mean()), float(group.std(ddof=0)))
            for key, group in usable.groupby(["metric_date", "sector"], sort=False)["cohort_ratio"]
        }

        out: list[Estimate] = []
        for position, row in enumerate(work.itertuples(index=False)):
            sector = getattr(row, "sector", None)
            if sector is None or pd.isna(sector):
                out.append(
                    self._refuse(
                        row, phash, "no configured sector, so there is no cohort to score within"
                    )
                )
                continue
            size, mean, sigma = stats.get((row.metric_date, sector), (0, 0.0, 0.0))
            if size < minimum:
                out.append(
                    self._refuse(
                        row,
                        phash,
                        f"cohort {sector!r} has {size} member(s) with a usable ratio on this "
                        f"date, below the minimum of {minimum}",
                    )
                )
                continue
            mine = ratios[position]
            if pd.isna(mine):
                out.append(
                    self._refuse(
                        row, phash, f"no usable {denominator_col} for this asset on this date"
                    )
                )
                continue
            if sigma <= 0:
                out.append(
                    self._refuse(
                        row,
                        phash,
                        f"cohort {sector!r} has no dispersion in its ratio, so a z-score is "
                        f"undefined",
                    )
                )
                continue
            z_score = (float(mine) - mean) / sigma
            ceiling = (size - 1) ** 0.5
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    # A z-score is not a price. returns_fair_value is false on the spec and
                    # fair_value is None on every row, so nothing downstream can average it
                    # into a valuation.
                    fair_value=None,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "cohort": sector,
                        "cohort_size": size,
                        "ratio": float(mine),
                        "cohort_mean": mean,
                        "cohort_stdev": sigma,
                        "numerator": numerator_col,
                        "denominator": denominator_col,
                        "window_rows": window,
                    },
                    diagnostics={
                        "z_score": z_score,
                        # A z-score over n members cannot exceed sqrt(n-1). When one member
                        # reaches that ceiling the others are compressed into a single point
                        # by construction -- measured 0.124 sigma apart on average -- and the
                        # score carries one bit ("that one is the outlier") rather than a
                        # ranking. True on 7.3% of scored days, and on 2026-10-13 itself.
                        "cohort_saturated": abs(z_score) > 0.99 * ceiling,
                        "status": "relative_score_not_a_price",
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out

    def _refuse(self, row, phash: str, reason: str) -> Estimate:
        return Estimate(
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
            reason=reason,
            panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
        )
