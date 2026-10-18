-- The volatility chain at grain (model_id, symbol, metric, window), spec 11.4.
--
-- FULL SAMPLE, by design: parameters are estimated over the whole history, so every
-- conditional volatility and correlation uses future data. No forecasting claim; not
-- comparable to fct_model_score.
--
-- symbol is an asset ('BTC') or a pair ('BTC|ETH'). window_name is NULL except on per-window
-- rows (the dissertation's correlation windows, config/volatility.yml).
-- model_id = '*' holds the hypothesis count: the family is every row whose metric is
-- 'p_value' (ARCH-LM, EGARCH gamma, the 6.3 exhaustion test, Engle-Sheppard).
-- A 'refused' row carries a NULL value and the measurement in reason.
select
    model_id,
    symbol,
    metric,
    value,
    n_rows,
    window_name,
    reason,
    params_hash,
    panel_vintage_ts
from read_parquet('{{ var("data_root") }}/gold/volatility_test/*.parquet')
