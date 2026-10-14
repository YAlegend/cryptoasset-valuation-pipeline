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

import math

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from cryptoval.methods.base import Estimate
from cryptoval.methods.benchmarks import fit_log_log
from cryptoval.methods.cost import _finite, _window_stats
from cryptoval.methods.monetary import _refuse
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


@register
class PetersonMetcalfe:
    spec = _SPECS["peterson_metcalfe"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("fit_window_rows", 365))
        min_fit = int(params.get("min_fit_rows", 180))
        phash = param_set_hash(params)
        addresses = pd.to_numeric(frame["active_addresses"], errors="coerce")
        pairs = addresses * (addresses - 1.0) / 2.0
        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")
        per_pair = (market_cap / pairs.where(pairs > 0)).where(market_cap > 0)
        per_pair = per_pair.where(np.isfinite(per_pair))
        # shift(1): the scale a row is valued with never contains that row's own market cap.
        prior = per_pair.shift(1)
        scale = prior.rolling(window, min_periods=min_fit).mean()
        _, usable = _window_stats(prior, window)

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            today_pairs, supply = _finite(pairs.iloc[position]), _finite(row.supply)
            fitted = _finite(scale.iloc[position])
            if fitted is None:
                out.append(
                    _refuse(
                        self.spec,
                        row,
                        params,
                        f"only {int(usable.iloc[position])} usable market-cap-per-pair "
                        f"observations in the last {window} rows, need {min_fit}",
                    )
                )
                continue
            if today_pairs is None or today_pairs <= 0 or supply is None or supply <= 0:
                out.append(
                    _refuse(
                        self.spec,
                        row,
                        params,
                        "no usable active_addresses or supply on this day "
                        f"(active_addresses={_finite(row.active_addresses)}, supply={supply})",
                    )
                )
                continue
            # The scale is the mean market cap per pair over the window: Peterson's constant,
            # FITTED because bitcoin-days-destroyed is not on the free tier.
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=fitted * today_pairs / supply,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "pairs": today_pairs,
                        "active_addresses": _finite(row.active_addresses),
                        "scale_constant": fitted,
                        "supply": supply,
                    },
                    diagnostics={
                        "fit_rows": int(usable.iloc[position]),
                        "window_rows": window,
                        "scale_constant_is_fitted": True,
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


@register
class NvmRatio:
    spec = _SPECS["nvm_ratio"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("fit_window_rows", 365))
        min_fit = int(params.get("min_fit_rows", 180))
        phash = param_set_hash(params)
        addresses = pd.to_numeric(frame["active_addresses"], errors="coerce")
        market_cap = pd.to_numeric(frame["market_cap_usd"], errors="coerce")
        both = pd.Series(
            np.where((addresses > 0) & (market_cap > 0) & np.isfinite(market_cap), 1.0, np.nan),
            index=frame.index,
        )
        _, usable = _window_stats(both.shift(1), window)

        out: list[Estimate] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            start = max(0, position - window)
            today_addresses = _finite(addresses.iloc[position])
            # The window ENDS at position, exclusive: no row sits on its own reference line.
            fit = fit_log_log(
                addresses.iloc[start:position],
                market_cap.iloc[start:position],
                min_fit,
                predict_at=today_addresses
                if today_addresses is not None and today_addresses > 0
                else None,
            )
            if fit is None:
                count = int(usable.iloc[position])
                reason = (
                    f"only {count} usable (positive, finite) rows in the last {window}, "
                    f"need {min_fit}"
                    if count < min_fit
                    else f"active_addresses has no variation across the {count} usable "
                    f"rows in the last {window}, so beta is not identified"
                )
                out.append(_refuse(self.spec, row, params, reason))
                continue
            today_cap = _finite(row.market_cap_usd)
            if (
                today_addresses is None
                or today_addresses <= 0
                or today_cap is None
                or today_cap <= 0
            ):
                out.append(
                    _refuse(
                        self.spec,
                        row,
                        params,
                        "no usable active_addresses or market cap on this day "
                        f"(active_addresses={today_addresses}, market_cap_usd={today_cap})",
                    )
                )
                continue
            implied = fit["fit_y"]
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    # None, not a number: declared returns_fair_value false.
                    fair_value=None,
                    band_low=None,
                    band_high=None,
                    inputs_used={
                        "active_addresses": today_addresses,
                        "market_cap_usd": today_cap,
                        "metcalfe_implied_cap": implied,
                    },
                    diagnostics={"nvm": today_cap / implied, **fit, "status": "deviation_series"},
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out


# fit_status values that are REFUSALS: the question could not be asked, so "no bubble" would
# be a claim the method has no basis for.
_LPPLS_REFUSED = frozenset({"window_too_short", "window_gaps", "flat_window", "optimiser_failed"})


_LPPLS_BOUND_MARGIN = 0.01  # of max_horizon_rows


def _lppls_fit(log_price: pd.Series, params: dict) -> dict:
    """Run the bounded multi-start LPPLS fit and say WHICH of its outcomes happened.

    ``fit_status`` is always set. ``bubble_regime`` is the only one that carries a date;
    ``tc_at_bound``, ``wrong_sign`` and ``poor_fit`` are optimiser results that were
    rejected (a real negative); the four in ``_LPPLS_REFUSED`` mean no verdict exists.
    Time is the row position inside the window, holes kept as holes, so a gap never
    compresses the axis the singularity is fitted on.
    """
    window = int(params.get("window_rows", 365))
    horizon = float(params.get("max_horizon_rows", 180))
    restarts = int(params.get("n_restarts", 8))
    result: dict = {"window_rows": window, "n_restarts": restarts}
    series = pd.to_numeric(log_price, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if len(series) < window:
        return {**result, "fit_status": "window_too_short", "detail": f"{len(series)} rows"}
    tail = series.iloc[-window:].to_numpy(dtype="float64")
    keep = np.isfinite(tail)
    need = math.ceil(float(params.get("min_window_coverage", 0.9)) * window)
    if int(keep.sum()) < need:
        return {
            **result,
            "fit_status": "window_gaps",
            "detail": f"only {int(keep.sum())} prices in the last {window} rows, need {need}",
        }
    values, t = tail[keep], np.flatnonzero(keep).astype("float64")
    spread = float(np.ptp(values))
    if spread <= 0:
        return {**result, "fit_status": "flat_window", "detail": "log price has zero range"}

    def solve(theta):
        """Linear parameters (a, b, c1, c2) by least squares given (tc, m, omega)."""
        tc, m, omega = theta
        dt = np.clip(tc - t, 1e-8, None)
        power, phase = dt**m, omega * np.log(dt)
        basis = np.column_stack(
            [np.ones_like(t), power, power * np.cos(phase), power * np.sin(phase)]
        )
        coef = np.linalg.lstsq(basis, values, rcond=None)[0]
        return coef, basis @ coef - values

    lower = [window + 1.0, 0.1, 6.0]
    upper = [window + horizon, 0.9, 13.0]
    rng = np.random.default_rng(0)  # Seeded: a bubble flag must not change between runs.
    best, converged, statuses = None, 0, []
    for _ in range(restarts):
        guess = [window + rng.uniform(1.0, horizon), rng.uniform(0.1, 0.9), rng.uniform(6.0, 13.0)]
        try:
            fit = least_squares(
                lambda theta: solve(theta)[1],
                guess,
                method="trf",
                bounds=(lower, upper),
                max_nfev=200,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            statuses.append(type(exc).__name__)
            continue
        statuses.append(int(fit.status))
        # Converged means scipy's own tolerance was met (status > 0); hitting max_nfev
        # (status 0) returns a point the optimiser was still moving from.
        if fit.success:
            converged += 1
            if best is None or fit.cost < best.cost:
                best = fit
    result["restarts_converged"] = converged
    if best is None:
        return {
            **result,
            "fit_status": "optimiser_failed",
            "detail": f"did not converge in any of {restarts} restarts (statuses {statuses})",
        }
    rmse_ratio = float(np.sqrt(2.0 * best.cost / len(values))) / spread
    tc, m, omega = (float(v) for v in best.x)
    b = float(solve(best.x)[0][1])
    result.update(best_rmse_ratio=rmse_ratio, tc=tc, m=m, omega=omega, b=b, rmse_ratio=rmse_ratio)
    # tc pinned to either bound is the optimiser saying "further (or sooner) than you let me look".
    # trf stops within its tolerance of a bound, not on it (a straight line ends 0.11 rows
    # short), so "at the bound" is within a margin rather than an equality.
    margin = _LPPLS_BOUND_MARGIN * horizon
    if tc >= window + horizon - margin or tc <= window + 1.0 + margin:
        return {**result, "fit_status": "tc_at_bound"}
    # B < 0 is the faster-than-exponential rise; B > 0 is a negative bubble or an anti-bubble.
    if b >= 0:
        return {**result, "fit_status": "wrong_sign"}
    if rmse_ratio > float(params.get("max_rmse_ratio", 0.02)):
        return {**result, "fit_status": "poor_fit"}
    return {**result, "fit_status": "bubble_regime"}


def fit_lppls(log_price: pd.Series, params: dict) -> dict | None:
    """The LPPLS fit of a log-price window, or None when there is no bubble regime to report.

    None covers every case with no date to report: a window too short or full of holes, an
    optimiser that never converged, tc landing on a bound of its declared horizon, a bubble
    term of the wrong sign, and a residual RMSE above ``max_rmse_ratio`` of the log-price
    range. A date from any of those would be a number invented by an optimiser. The method
    itself uses ``_lppls_fit`` so it can tell those cases apart; None here is the summary.
    """
    fit = _lppls_fit(log_price, params)
    if fit["fit_status"] != "bubble_regime":
        return None
    return {k: fit[k] for k in ("tc", "m", "omega", "rmse_ratio", "window_rows")}


@register
class LpplsBubbleFlag:
    spec = _SPECS["lppls_bubble_flag"]

    def estimate(self, frame: pd.DataFrame, params: dict) -> list[Estimate]:
        window = int(params.get("window_rows", 365))
        stride = max(1, int(params.get("stride_rows", 1)))
        phash = param_set_hash(params)
        price = pd.to_numeric(frame["price_usd"], errors="coerce")
        log_price = np.log(price.where(price > 0))

        out: list[Estimate] = []
        carried: dict | None = None  # the last fit and the position it was made at
        fitted_at = -1
        for position, row in enumerate(frame.itertuples(index=False)):
            if position < window:
                out.append(
                    _refuse(
                        self.spec,
                        row,
                        params,
                        f"only {position} prior rows; the fit window is {window}",
                    )
                )
                continue
            # tc counted from the row being valued: the window ends one row before it.
            remaining = (
                None if carried is None else fitted_at + carried.get("tc", 0) - window - position
            )
            expired = (
                carried is not None and carried["fit_status"] == "bubble_regime" and remaining < 1
            )
            if carried is None or position - fitted_at >= stride or expired:
                # iloc[:position]: the window ends before this row, prior-only.
                carried, fitted_at = _lppls_fit(log_price.iloc[:position], params), position
            fit, age = carried, position - fitted_at
            if fit["fit_status"] in _LPPLS_REFUSED:
                out.append(_refuse(self.spec, row, params, f"{fit['fit_status']}: {fit['detail']}"))
                continue
            detected = fit["fit_status"] == "bubble_regime"
            out.append(
                Estimate(
                    symbol=row.symbol,
                    metric_date=row.metric_date,
                    method_id=self.spec.method_id,
                    param_set_hash=phash,
                    fair_value=None,
                    band_low=None,
                    band_high=None,
                    inputs_used={"window_rows": window, "n_restarts": fit["n_restarts"]},
                    diagnostics={
                        "bubble_regime": detected,
                        "fit_status": fit["fit_status"],
                        # A date ONLY when the fit survived every gate; otherwise None, not
                        # the optimiser's best guess. Rows from this row's own date.
                        "critical_time": (fitted_at + fit["tc"] - window - position)
                        if detected
                        else None,
                        "m": fit["m"] if detected else None,
                        "omega": fit["omega"] if detected else None,
                        "rmse_ratio": fit["rmse_ratio"] if detected else None,
                        "best_rmse_ratio": fit["best_rmse_ratio"],
                        "restarts_converged": fit["restarts_converged"],
                        "restarts_attempted": fit["n_restarts"],
                        "fit_age_rows": age,
                    },
                    applicable=True,
                    panel_vintage_ts=getattr(row, "panel_vintage_ts", None),
                )
            )
        return out
