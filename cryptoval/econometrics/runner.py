"""One long table for the whole anchoring chain (spec 11.4), at grain
(pair_id, symbol, metric).

The same contract as fct_model_score: a refusal is a ROW carrying the measurement that
caused it, not a silently absent asset, because which assets could not be tested is a
finding this chapter has to report.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import select_order

from cryptoval.econometrics.base import PairSpec, Unusable
from cryptoval.econometrics.cointegration import engle_granger, johansen
from cryptoval.econometrics.series import pair_rows
from cryptoval.econometrics.stationarity import adf, kpss
from cryptoval.econometrics.vecm import speed_of_adjustment
from cryptoval.hashing import stable_hash
from cryptoval.scoring.multiplicity import ALPHA, correct_family_wise

TEST_COLUMNS = [
    "pair_id",
    "symbol",
    "metric",
    "value",
    "n_rows",
    "lags",
    "reason",
    "params_hash",
    "panel_vintage_ts",
]

COUNT_KEYS = {"pair_id": "*", "symbol": "*"}

# What a singular or degenerate system raises from statsmodels; each becomes a refusal row.
SYSTEM_FIT_ERRORS = (np.linalg.LinAlgError, ValueError)

# The largest VAR order the criterion may choose: two weeks of daily lags.
MAX_VAR_LAGS = 14


def _lag_order(pair: pd.DataFrame, criterion: str) -> int:
    """Design 9: the lag order comes from an information criterion, and is recorded."""
    levels = pair[["y", "x"]].to_numpy(dtype="float64")
    chosen = select_order(levels, maxlags=MAX_VAR_LAGS, deterministic="ci")
    return int(getattr(chosen, criterion.lower()))


def _row(**values: Any) -> dict[str, Any]:
    row = dict.fromkeys(TEST_COLUMNS)
    row.update(values)
    return row


def run_pairs(
    levels: pd.DataFrame,
    specs: dict[str, PairSpec],
    panel_vintage_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Every pair against every symbol in ``levels``, full sample."""
    rows: list[dict[str, Any]] = []
    tested: dict[tuple[str, str], tuple[pd.DataFrame, dict[str, Any], int]] = {}

    for pair_id, spec in sorted(specs.items()):
        # The WHOLE spec, not spec.params: the three pairs share identical params, and an
        # edit to independent or forced_coefficient must change the hash too.
        params_hash = stable_hash(dataclasses.asdict(spec))
        criterion = spec.params.get("lag_criterion", "aic")
        common = {
            "pair_id": pair_id,
            "params_hash": params_hash,
            "panel_vintage_ts": panel_vintage_ts,
        }

        for symbol in sorted(levels["symbol"].unique()):
            try:
                pair = pair_rows(levels, str(symbol), spec)
            except Unusable as exc:
                rows.append(_row(symbol=str(symbol), metric="refused", reason=str(exc), **common))
                continue

            n = float(len(pair))
            # BOTH series, not just the dependent one: cointegration is a statement about
            # two I(1) series, so a reader cannot judge the finding without the
            # independent series' unit-root evidence too.
            for name, series in (
                ("", pair["y"]),
                ("independent_", pair["x"]),
                ("ratio_", pair["ratio"]),
            ):
                for test, fn in (
                    ("adf", functools.partial(adf, lag_criterion=criterion)),
                    ("kpss", kpss),
                ):
                    try:
                        result = fn(series)
                    except Unusable as exc:
                        rows.append(
                            _row(
                                symbol=str(symbol),
                                metric=f"{name}{test}_statistic",
                                reason=str(exc),
                                n_rows=n,
                                **common,
                            )
                        )
                        continue
                    for key in ("statistic", "p_value"):
                        rows.append(
                            _row(
                                symbol=str(symbol),
                                metric=f"{name}{test}_{key}",
                                value=result[key],
                                n_rows=result["n"],
                                lags=result["lags"],
                                **common,
                            )
                        )

            try:
                eg = engle_granger(pair["y"], pair["x"])
            except Unusable as exc:
                rows.append(
                    _row(
                        symbol=str(symbol),
                        metric="cointegration_statistic",
                        reason=str(exc),
                        n_rows=n,
                        **common,
                    )
                )
                continue

            for key, metric in (
                ("statistic", "cointegration_statistic"),
                ("p_value", "cointegration_p_value"),
                ("coefficient", "cointegration_coefficient"),
                ("coefficient_stderr", "cointegration_coefficient_stderr"),
            ):
                rows.append(
                    _row(symbol=str(symbol), metric=metric, value=eg[key], n_rows=eg["n"], **common)
                )

            # A system fit that raises becomes a refusal row: one singular pair must not lose
            # the whole run, and the Engle-Granger claim above stands without it.
            lags: int | None = None
            try:
                lags = _lag_order(pair, criterion)
                jo = johansen(pair["y"], pair["x"], lags=lags)
            except SYSTEM_FIT_ERRORS as exc:
                rows.append(
                    _row(
                        symbol=str(symbol),
                        metric="johansen_trace_statistic",
                        reason=f"johansen raised {type(exc).__name__}: {exc}",
                        n_rows=n,
                        **common,
                    )
                )
            else:
                for key, metric in (
                    ("trace_statistic", "johansen_trace_statistic"),
                    ("critical_95", "johansen_critical_95"),
                    ("rank", "johansen_rank"),
                ):
                    rows.append(
                        _row(
                            symbol=str(symbol),
                            metric=metric,
                            value=jo[key],
                            n_rows=jo["n"],
                            lags=jo["lags"],
                            **common,
                        )
                    )

            tested[(pair_id, str(symbol))] = (pair, common, lags)

    table = correct_family_wise(
        pd.DataFrame(rows, columns=TEST_COLUMNS),
        p_metric="cointegration_p_value",
        columns=TEST_COLUMNS,
        count_keys={
            **COUNT_KEYS,
            "params_hash": stable_hash({p: dataclasses.asdict(s) for p, s in specs.items()}),
            "panel_vintage_ts": panel_vintage_ts,
        },
    )

    # Design 7: the VECM runs AFTER the correction, on survivors only. A speed of return to
    # an equilibrium the correction just rejected would read as a finding.
    holm = table[table["metric"] == "cointegration_p_value_holm"]
    adjusted = holm.set_index(["pair_id", "symbol"])["value"].to_dict()
    vecm_rows: list[dict[str, Any]] = []
    for (pair_id, symbol), (pair, common, lags) in tested.items():
        p_holm = adjusted[(pair_id, symbol)]
        if not p_holm < ALPHA:
            vecm_rows.append(
                _row(
                    symbol=symbol,
                    metric="vecm_alpha",
                    n_rows=float(len(pair)),
                    reason=f"vecm: not fitted, cointegration_p_value_holm "
                    f"{p_holm:.3g} is not below {ALPHA}",
                    **common,
                )
            )
            continue
        try:
            if lags is None:
                raise ValueError("no lag order, the VAR lag selection itself failed")
            vecm = speed_of_adjustment(pair["y"], pair["x"], lags=lags)
        except SYSTEM_FIT_ERRORS as exc:
            vecm_rows.append(
                _row(
                    symbol=symbol,
                    metric="vecm_alpha",
                    n_rows=float(len(pair)),
                    reason=f"vecm raised {type(exc).__name__}: {exc}",
                    **common,
                )
            )
            continue
        for key, metric in (
            ("alpha", "vecm_alpha"),
            ("alpha_stderr", "vecm_alpha_stderr"),
            ("alpha_independent", "vecm_alpha_independent"),
            ("alpha_independent_stderr", "vecm_alpha_independent_stderr"),
            ("gap_speed", "vecm_gap_speed"),
            ("half_life_days", "vecm_half_life_days"),
        ):
            vecm_rows.append(
                _row(
                    symbol=symbol,
                    metric=metric,
                    value=vecm[key],
                    n_rows=vecm["n"],
                    lags=vecm["lags"],
                    **common,
                )
            )

    if not vecm_rows:
        return table
    return pd.concat([table, pd.DataFrame(vecm_rows, columns=TEST_COLUMNS)], ignore_index=True)
