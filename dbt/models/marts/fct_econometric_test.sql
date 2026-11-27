-- The anchoring chain at grain (pair_id, symbol, metric), spec 11.4.
--
-- FULL SAMPLE, by design. Every statistic here is computed over the whole available
-- history and therefore uses future data; this table supports no forecasting claim and is
-- not comparable to fct_model_score, which is strictly out of sample.
--
-- pair_id = '*' holds the hypothesis count the family-wise correction was taken over, and
-- that family is the cointegration tests ONLY: ADF/KPSS are pre-tests that decide which
-- regressions are valid, and VECM coefficients estimate an already-established
-- relationship.
--
-- A 'refused' row carries a NULL value and the measurement in reason, the same contract
-- fct_valuation_estimate and fct_model_score use.
select
    pair_id,
    symbol,
    metric,
    value,
    n_rows,
    lags,
    reason,
    params_hash,
    panel_vintage_ts
from read_parquet('{{ var("data_root") }}/gold/econometric_test/*.parquet')
