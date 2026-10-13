"""Family A -- monetary / equation of exchange (spec 10.2).

The holding-time correction is the load-bearing piece. If velocity is log-normal, mean
holding time is E[1/V] = exp(-mu + sigma^2/2) while the naive inversion 1/E[V] gives
exp(-mu - sigma^2/2): the two differ by exactly exp(sigma^2), and the dissertation's QTM
and Price-to-Utility sections use the inversion. On a realistic sigma of 0.6 that is a 43%
error in the denominator, in a known direction.

The spec says to derive the mean by simulation; the closed form above is exact, so it is
used here and Monte Carlo verifies it in the tests instead. Adding sampling noise to a
quantity we can state precisely would be a strange choice.

On QTM's circularity: the equation wants Q as on-chain transaction VALUE, which the free
tier does not serve. Using exchange volume makes V = volume/market_cap, and the whole
expression collapses to price/f -- it returns the market price back. Rather than hide
that, both readings ship: crypto_qtm takes velocity as an exogenous assumption and is a
genuine sensitivity, and crypto_qtm_panel_velocity uses the panel and is flagged circular
with collapses_to_price in its diagnostics. Where they differ IS the finding (spec 14).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import normal_ad

from cryptoval.methods.base import Estimate, MethodSpec
from cryptoval.methods.registry import (
    load_method_specs,
    param_set_hash,
    register,
    register_instance,
)

_SPECS = load_method_specs()


def holding_time_ratio(sigma: float) -> float:
    """E[1/V] / (1/E[V]) for a log-normal V. Exactly exp(sigma^2)."""
    return float(math.exp(sigma**2))


def fit_lognormal_velocity(series: pd.Series) -> dict:
    """Fit log(V) ~ Normal and return both holding-time constructions.

    Only strictly positive velocities enter the fit: a zero or negative velocity is not a
    draw from a log-normal, it is a day with no measurable turnover.
    """
    values = pd.to_numeric(series, errors="coerce")
    positive = values[values > 0].dropna()
    if len(positive) < 30:
        raise ValueError(
            f"only {len(positive)} strictly positive velocity observations; "
            "need at least 30 to fit a distribution"
        )
    logs = np.log(positive.to_numpy())
    mu = float(logs.mean())
    sigma = float(logs.std(ddof=1))
    try:
        _, ad_p = normal_ad(logs)
    except Exception:
        ad_p = float("nan")
    return {
        "mu": mu,
        "sigma": sigma,
        "n": int(len(positive)),
        "mean_velocity": float(math.exp(mu + sigma**2 / 2)),
        "mean_holding_time": float(math.exp(-mu + sigma**2 / 2)),
        "naive_inverse_velocity": float(math.exp(-mu - sigma**2 / 2)),
        "holding_time_ratio": holding_time_ratio(sigma),
        # p < 0.05 rejects log-normality, which does not invalidate the ratio but does
        # mean the closed form is an approximation for this asset. Reported, not hidden.
        "anderson_darling_p": float(ad_p),
    }


def _refuse(spec: MethodSpec, row, params: dict, reason: str) -> Estimate:
    return Estimate(
        symbol=row.symbol,
        metric_date=row.metric_date,
        method_id=spec.method_id,
        param_set_hash=param_set_hash(params),
        fair_value=None,
        band_low=None,
        band_high=None,
        inputs_used={},
        diagnostics={},
        applicable=False,
        reason=reason,
        panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
    )


@register
class HoldingTimeCorrection:
    spec = _SPECS["holding_time_correction"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        out: list[Estimate] = []
        phash = param_set_hash(params)
        for position, row in enumerate(frame.itertuples(index=False)):
            # Trailing window only: a fit that saw future velocity would leak into
            # Plan 4's backtests.
            history = frame["velocity_total"].iloc[max(0, position - window + 1) : position + 1]
            try:
                fit = fit_lognormal_velocity(history)
            except ValueError as exc:
                out.append(_refuse(self.spec, row, params, f"no positive velocity: {exc}"))
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    # A correction, not a valuation: returns_fair_value is false in config.
                    fair_value=None,
                    band_low=None,
                    band_high=None,
                    inputs_used={"window_rows": window, "observations": fit["n"]},
                    diagnostics=fit,
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


def _velocity_for(row, params: dict) -> tuple[float | None, bool]:
    """Resolve the velocity term. Returns (value, collapses_to_price)."""
    if params.get("velocity_source") == "panel":
        value = getattr(row, "velocity_total", None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None, True
        return (value if math.isfinite(value) and value > 0 else None), True
    value = float(params["velocity_daily"])
    return (value if value > 0 else None), False


class CryptoQTM:
    """price = Q / (M* x f x V), dissertation section 3.1."""

    def __init__(self, spec: MethodSpec) -> None:
        self.spec = spec

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        fraction = float(params.get("circulating_fraction", 1.0))
        phash = param_set_hash(params)
        out: list[Estimate] = []
        for row in frame.itertuples(index=False):
            velocity, collapses = _velocity_for(row, params)
            quantity = pd.to_numeric(pd.Series([row.volume_total_usd]), errors="coerce").iloc[0]
            supply = float(row.supply) if row.supply else None
            if velocity is None or supply is None or supply <= 0 or not np.isfinite(quantity):
                out.append(_refuse(self.spec, row, params, "missing volume, supply or velocity"))
                continue
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=float(quantity) / (supply * fraction * velocity),
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "volume_total_usd": float(quantity),
                        "supply": supply,
                        "circulating_fraction": fraction,
                        "velocity": velocity,
                        "velocity_source": params.get("velocity_source", "parameter"),
                        "price_usd": float(row.price_usd) if row.price_usd else None,
                    },
                    diagnostics={
                        # True means this row IS the market price divided by f, so its
                        # residual against price is zero by construction.
                        "collapses_to_price": collapses,
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


register_instance(CryptoQTM(_SPECS["crypto_qtm"]))
register_instance(CryptoQTM(_SPECS["crypto_qtm_panel_velocity"]))


@register
class JCurve:
    """Burniske's CUV/DEUV split.

    CUV = network transaction value / (circulating supply x velocity); DEUV = price - CUV.
    The dissertation renders CUV as "token's current supply / (token supply x velocity)",
    which is dimensionally incoherent as printed -- it yields a ratio, not a price. We
    implement the economically intended form and the method card records the deviation
    (spec 10.2).
    """

    spec = _SPECS["j_curve"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        phash = param_set_hash(params)
        out: list[Estimate] = []
        for row in frame.itertuples(index=False):
            velocity, collapses = _velocity_for(row, params)
            supply = float(row.supply) if row.supply else None
            price = float(row.price_usd) if row.price_usd else None
            quantity = float(row.volume_total_usd) if row.volume_total_usd else None
            if None in (velocity, supply, price, quantity) or supply <= 0:
                out.append(
                    _refuse(self.spec, row, params, "missing volume, supply, price or velocity")
                )
                continue
            cuv = quantity / (supply * velocity)
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=cuv,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "volume_total_usd": quantity,
                        "supply": supply,
                        "velocity": velocity,
                        "price_usd": price,
                    },
                    diagnostics={
                        "deuv": price - cuv,
                        "deuv_share": (price - cuv) / price if price else None,
                        "collapses_to_price": collapses,
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out
