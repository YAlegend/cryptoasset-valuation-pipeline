"""How often Engle-Granger and Johansen reject on THIS panel when no equilibrium exists.

The null: each pair's fundamental has its daily differences circularly shifted against market
cap's, then re-integrated. Each series keeps its own autocorrelation, fat tails and volatility
clustering, and the link between the two is gone. The runner's own functions run on the result,
AIC lag rule and the 14-lag cap included, so the size is measured at the lags that actually run.

shortcut: independent walks are one null; two series sharing a crypto-wide trend without
cointegrating are another, not simulated. Run: python -m cryptoval.econometrics.null_size [reps]
"""

from __future__ import annotations

import sys
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor

import duckdb
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from cryptoval.cli import _read_econ_inputs
from cryptoval.config import load_config
from cryptoval.econometrics.base import Unusable
from cryptoval.econometrics.cointegration import engle_granger, johansen
from cryptoval.econometrics.registry import PAIRS
from cryptoval.econometrics.runner import MAX_VAR_LAGS, SYSTEM_FIT_ERRORS, _lag_order
from cryptoval.econometrics.series import build_levels, pair_rows
from cryptoval.scoring.multiplicity import ALPHA


def _pairs(levels: pd.DataFrame) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    out = {}
    for pair_id, spec in sorted(PAIRS.items()):
        for symbol in sorted(levels["symbol"].unique()):
            try:
                pair = pair_rows(levels, str(symbol), spec)
                engle_granger(pair["y"], pair["x"])
            except Unusable:
                continue
            out[(pair_id, str(symbol))] = (pair["y"].to_numpy(float), pair["x"].to_numpy(float))
    return out


def _draw(job: tuple) -> tuple:
    (pair_id, symbol), rep, y, x = job
    if rep >= 0:  # rep -1 is the live pair, unshifted: the check that the wiring is the runner's
        rng = np.random.default_rng(zlib.crc32(f"{pair_id}{symbol}{rep}".encode()))
        shift = int(rng.integers(len(x) // 4, 3 * len(x) // 4))
        x = np.concatenate([[x[0]], x[0] + np.cumsum(np.roll(np.diff(x), shift))])
    # Here, not in main: spawned workers do not inherit the parent's filters.
    warnings.filterwarnings("ignore")
    ys, xs = pd.Series(y), pd.Series(x)
    p_value = engle_granger(ys, xs)["p_value"]
    try:
        lags = _lag_order(pd.DataFrame({"y": ys, "x": xs}), "aic")
        rank = johansen(ys, xs, lags=lags)["rank"]
    except SYSTEM_FIT_ERRORS:
        lags, rank = -1, np.nan
    return pair_id, symbol, rep, p_value, lags, rank


def main(replicates: int = 100) -> None:
    warnings.filterwarnings("ignore")
    settings, _ = load_config()
    pairs = _pairs(build_levels(_read_econ_inputs(settings)))
    jobs = [(k, rep, y, x) for k, (y, x) in pairs.items() for rep in range(-1, replicates)]
    with ProcessPoolExecutor() as pool:
        draws = pd.DataFrame(
            list(pool.map(_draw, jobs, chunksize=8)),
            columns=["pair_id", "symbol", "rep", "eg_p", "lags", "rank"],
        )

    live = draws[draws["rep"] == -1].set_index(["pair_id", "symbol"])
    con = duckdb.connect(f"{settings.data_root}/cryptoval.duckdb", read_only=True)
    try:
        mart = con.execute(
            "select pair_id, symbol, value from fct_econometric_test "
            "where metric = 'cointegration_p_value'"
        ).fetchdf()
    finally:
        con.close()
    gap = (live["eg_p"] - mart.set_index(["pair_id", "symbol"])["value"]).abs().max()
    assert gap < 1e-12, f"the unshifted pairs differ from the mart by {gap}: rerun econ first"

    null = draws[draws["rep"] >= 0].assign(eg=lambda d: d["eg_p"] < ALPHA)
    holm_bar = ALPHA / len(pairs)
    print(f"{len(pairs)} pairs x {replicates} null draws")
    print(f"size at {ALPHA}: Engle-Granger {null['eg'].mean():.3f}, ", end="")
    print(f"Johansen {null['rank'].mean():.3f}")
    bins = pd.cut(null["lags"], [-2, -0.5, 2, MAX_VAR_LAGS - 1, MAX_VAR_LAGS])
    print(null.groupby(bins, observed=True)[["eg", "rank"]].agg(["size", "mean"]).round(3))
    print(null.groupby("pair_id")[["eg", "rank"]].mean().round(3))

    survivors = live.index[multipletests(live["eg_p"], alpha=ALPHA, method="holm")[0]]
    by_pair = null.groupby(["pair_id", "symbol"])["eg_p"]
    table = pd.DataFrame(
        {
            "live_p": live.loc[survivors, "eg_p"],
            "null_below_alpha": by_pair.apply(lambda p: (p < ALPHA).mean()).loc[survivors],
            "null_below_holm_bar": by_pair.apply(lambda p: (p < holm_bar).mean()).loc[survivors],
            "null_min_p": by_pair.min().loc[survivors],
        }
    )
    print(f"\nlive Holm survivors against their own null (Holm bar {holm_bar:.1e}):")
    print(table.sort_values("null_below_holm_bar").to_string(float_format="{:.3g}".format))

    false = [
        g.loc[multipletests(g["eg_p"], alpha=ALPHA, method="holm")[0], "symbol"].tolist()
        for _, g in null.groupby("rep")
    ]
    print(f"\nHolm over the family under the null: FWER {np.mean([bool(f) for f in false]):.2f}")
    print(pd.Series([s for f in false for s in f]).value_counts().to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
