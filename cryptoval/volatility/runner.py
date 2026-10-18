"""One long table for the volatility chain, at grain (model_id, symbol, metric, window).

The same contract as fct_econometric_test: a refusal is a ROW carrying the measurement,
not a silently absent asset. ONE Holm family over every row whose metric is 'p_value':
ARCH-LM, EGARCH gamma, the 6.3 exhaustion test, Engle-Sheppard. Everything else is an
estimate or a description (design 11).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from cryptoval.econometrics.base import Unusable
from cryptoval.hashing import stable_hash
from cryptoval.scoring.multiplicity import correct_family_wise
from cryptoval.volatility.dcc import engle_sheppard, fit_dcc
from cryptoval.volatility.exhaustion import exhaustion_test
from cryptoval.volatility.garch import arch_lm, fit_egarch, fit_garch
from cryptoval.volatility.regimes import REGIME_SCALARS, fit_regimes
from cryptoval.volatility.returns import crypto_returns, pair_returns, price_series
from cryptoval.volatility.settings import VolConfig

TEST_COLUMNS = [
    "model_id",
    "symbol",
    "metric",
    "value",
    "n_rows",
    "window_name",
    "reason",
    "params_hash",
    "panel_vintage_ts",
]

COUNT_KEYS = {"model_id": "*", "symbol": "*"}

# Below this a window's mean is a handful of days, and the reason names the count.
MIN_WINDOW_ROWS = 20

# What a singular or degenerate fit raises from numpy/scipy/arch; each becomes a refusal.
FIT_ERRORS = (Unusable, np.linalg.LinAlgError, ValueError)

GARCH_KEYS = ("omega", "alpha", "beta", "nu", "persistence", "half_life_days")
DCC_KEYS = ("a", "b", "persistence")


def _row(**values: Any) -> dict[str, Any]:
    row = dict.fromkeys(TEST_COLUMNS)
    row.update(values)
    return row


def _in_window(index: pd.DatetimeIndex, window: tuple) -> np.ndarray:
    start, end = window
    return np.asarray((index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end)))


def run_volatility(
    crypto: pd.DataFrame,
    traditional: pd.DataFrame,
    config: VolConfig,
    stablecoins: frozenset[str] = frozenset(),
    panel_vintage_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    common = {
        "params_hash": stable_hash(dataclasses.asdict(config)),
        "panel_vintage_ts": panel_vintage_ts,
    }
    rows: list[dict[str, Any]] = []

    def refuse(model_id: str, symbol: str, reason: Any, n: float | None = None) -> None:
        rows.append(
            _row(
                model_id=model_id,
                symbol=symbol,
                metric="refused",
                reason=str(reason),
                n_rows=n,
                **common,
            )
        )

    def emit(model_id: str, symbol: str, values: dict, keys: tuple, n: float) -> None:
        for key in keys:
            value = float(values[key])
            reason = None
            if key == "half_life_days" and np.isnan(value):
                reason = f"persistence {values['persistence']:.4f} >= 1: no half life"
            rows.append(
                _row(
                    model_id=model_id,
                    symbol=symbol,
                    metric=key,
                    value=value,
                    n_rows=n,
                    reason=reason,
                    **common,
                )
            )

    def per_window(model_id, symbol, metric, index, compute) -> None:
        for name, window in sorted(config.windows.items()):
            mask = _in_window(index, window)
            k = int(mask.sum())
            if k < MIN_WINDOW_ROWS:
                rows.append(
                    _row(
                        model_id=model_id,
                        symbol=symbol,
                        metric=metric,
                        window_name=name,
                        n_rows=float(k),
                        reason=f"{k} rows in window {name}, {MIN_WINDOW_ROWS} needed",
                        **common,
                    )
                )
                continue
            rows.append(
                _row(
                    model_id=model_id,
                    symbol=symbol,
                    metric=metric,
                    window_name=name,
                    value=float(compute(mask)),
                    n_rows=float(k),
                    **common,
                )
            )

    prices: dict[str, pd.Series] = {}
    usable: list[str] = []
    for symbol in sorted(str(s) for s in crypto["symbol"].unique()):
        if symbol in stablecoins:
            refuse(
                "returns",
                symbol,
                f"{symbol} is a stablecoin: a pegged price's variance is not a market quantity",
            )
            continue
        try:
            prices[symbol] = price_series(crypto, symbol)
            returns = crypto_returns(prices[symbol], symbol)
        except Unusable as exc:
            refuse("returns", symbol, exc)
            continue
        usable.append(symbol)
        n = float(len(returns))

        try:
            emit(
                "arch_lm",
                symbol,
                arch_lm(returns, config.arch_lm_lags),
                ("statistic", "p_value"),
                n,
            )
        except FIT_ERRORS as exc:
            refuse("arch_lm", symbol, exc, n)

        garch = None
        try:
            garch = fit_garch(returns)
            emit("garch", symbol, garch, GARCH_KEYS, n)
        except FIT_ERRORS as exc:
            refuse("garch", symbol, exc, n)

        try:
            emit("egarch", symbol, fit_egarch(returns), ("gamma", "gamma_stderr", "p_value"), n)
        except FIT_ERRORS as exc:
            refuse("egarch", symbol, exc, n)

        if garch is None:
            refuse("exhaustion", symbol, "exhaustion: needs the garch fit, which was refused", n)
        else:
            try:
                result = exhaustion_test(returns, garch["cond_vol"], config.exhaustion_block_days)
                emit(
                    "exhaustion",
                    symbol,
                    result,
                    ("p_value", "continuation_high_vol", "continuation_low_vol", "block_pairs"),
                    n,
                )
            except FIT_ERRORS as exc:
                refuse("exhaustion", symbol, exc, n)

        try:
            regimes = fit_regimes(returns, search_reps=config.markov_search_reps)
        except FIT_ERRORS as exc:
            refuse("markov", symbol, exc, n)
        else:
            emit("markov", symbol, regimes, REGIME_SCALARS, n)
            probability = regimes["turbulent_probability"]
            per_window(
                "markov",
                symbol,
                "turbulent_probability",
                probability.index,
                lambda mask, p=probability: p[mask].mean(),
            )

    pairs: list[tuple[str, str, bool]] = []
    for anchor in config.crypto_anchors:
        pairs += [(anchor, other, False) for other in usable if other != anchor]
    for anchor in config.traditional_anchors:
        pairs += [(anchor, s, True) for s in sorted(str(s) for s in traditional["symbol"].unique())]

    for left, right, trading in pairs:
        label = f"{left}|{right}"
        try:
            left_prices = prices[left] if left in prices else price_series(crypto, left)
            right_prices = (
                price_series(traditional, right)
                if trading
                else prices[right]
                if right in prices
                else price_series(crypto, right)
            )
            pair = pair_returns(left_prices, right_prices, label, trading_calendar=trading)
            z = np.column_stack(
                [
                    fit_garch(pair["a"])["std_resid"].to_numpy(),
                    fit_garch(pair["b"])["std_resid"].to_numpy(),
                ]
            )
            dcc = fit_dcc(z)
            es = engle_sheppard(z, config.engle_sheppard_lags)
        except FIT_ERRORS as exc:
            refuse("dcc", label, exc)
            continue

        n = float(len(pair))
        rho = pd.Series(dcc["rho"], index=pair.index)
        emit("dcc", label, dcc, DCC_KEYS, n)
        emit(
            "dcc",
            label,
            {
                "correlation_mean": rho.mean(),
                "correlation_min": rho.min(),
                "correlation_max": rho.max(),
            },
            ("correlation_mean", "correlation_min", "correlation_max"),
            n,
        )
        emit(
            "dcc",
            label,
            {"engle_sheppard_statistic": es["statistic"], "p_value": es["p_value"]},
            ("engle_sheppard_statistic", "p_value"),
            n,
        )
        per_window("dcc", label, "dcc_correlation", pair.index, lambda m, r=rho: r[m].mean())
        per_window(
            "dcc",
            label,
            "pearson_correlation",
            pair.index,
            lambda m, p=pair: p["a"][m].corr(p["b"][m]),
        )

        # Descriptive, not a test: rolling means overlap, so no rank has a p-value.
        peak = _in_window(pair.index, config.windows[config.peak_window])
        k = int(peak.sum())
        if k < MIN_WINDOW_ROWS:
            rows.append(
                _row(
                    model_id="dcc",
                    symbol=label,
                    metric="peak_percentile",
                    window_name=config.peak_window,
                    n_rows=float(k),
                    reason=f"{k} rows in window {config.peak_window}, {MIN_WINDOW_ROWS} needed",
                    **common,
                )
            )
        else:
            rolling = rho.rolling(k).mean().dropna()
            rank = float((rolling < rho[peak].mean()).mean())
            rows.append(
                _row(
                    model_id="dcc",
                    symbol=label,
                    metric="peak_percentile",
                    window_name=config.peak_window,
                    value=rank,
                    n_rows=float(len(rolling)),
                    **common,
                )
            )

    return correct_family_wise(
        pd.DataFrame(rows, columns=TEST_COLUMNS),
        p_metric="p_value",
        columns=TEST_COLUMNS,
        count_keys={**COUNT_KEYS, **common},
    )
