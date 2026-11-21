# lppls_bubble_flag

**Family:** B_network
**Citation:** Sornette LPPLS; Wheatley et al. (2018) applied to cryptoassets (spec 10.3)
**Returns a fair value:** no, it is a correction

## Data dependencies

- `price_usd`

Minimum coverage density: **90%** per column. Minimum observations:
**500**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

- `max_horizon_rows` = `180`
- `max_rmse_ratio` = `0.02`
- `min_restarts_converged` = `4`
- `min_window_coverage` = `0.9`
- `n_restarts` = `8`
- `stride_rows` = `5`
- `window_rows` = `365`

## Failure modes

- MEASURED NEGATIVE on this panel: at a 365-row window the flag fires on 0 of 55,125 applicable rows. The binding constraint is fit fidelity, not the bounds -- the best converged fit anywhere has a residual RMSE of 3.34% of its window's log range against the 2% requirement, the median is 9.7%, and no restart on any row passes every gate. Relaxing the floor to 5% would fire on 690 rows and to 10% on 6,586, so the floor IS the flag and this panel supports no value of it. Every row publishes best_rmse_ratio, b, fit_status and restarts_passing_gates so a reader can apply their own floor rather than inherit this one.
- The optimiser converges on SOMETHING for almost any smooth series. The bound on tc, the sign of the bubble term and the RMSE floor exist to turn that into a "no bubble regime detected"; without them this method emits a crash date for every asset every day.
- Three different things are NOT the same and are kept apart on every row. fit_status says which: bubble_regime (all gates passed); tc_at_bound, wrong_sign or poor_fit (an optimiser result that was rejected - applicable, no bubble); and, as a REFUSAL rather than a negative, optimiser_failed (no restart converged), window_gaps and flat_window. A non-converging optimiser is never reported as "no bubble".
- A detected bubble regime is not a forecast: LPPLS identifies a faster-than-exponential pattern, and the literature's own out-of-sample record on crypto is contested.
- The fit is on a fixed trailing window, so a regime that began before the window opens is invisible.
- Multi-start optimisation makes the result depend on the restart count; n_restarts travels on every row for reproducibility, and the restarts are seeded.
- The RMSE floor is tight against daily crypto noise, so the flag may fire rarely or never on an asset; restarts_converged and best_rmse_ratio travel on every row so a reader can see how close a rejected fit came.
- The fit runs every stride_rows rows and carries its verdict between; fit_age_rows says how stale.

## Assumptions

- A bubble is a faster-than-exponential price path with log-periodic oscillations.
- 365 rows is long enough to contain such a regime and short enough to be one regime.
