-- Walk-forward results at spec 8.1's grain (model_id, asset, fold, horizon, metric).
-- The naive baseline is stored as just another model_id, so "did it beat the random
-- walk" is a query rather than a claim.
--
-- fold = -1 marks a value pooled across every fold: the Diebold-Mariano statistic, its
-- p-value, and the Holm-adjusted p-value live there. horizon = -1 with model_id = '*'
-- is the hypothesis count the family-wise correction was taken over (spec 11.6).
--
-- A 'refused' row carries a NULL value and the measurement in reason, the same contract
-- fct_valuation_estimate uses: a refusal with its measurement is a research finding.
select
    model_id,
    symbol,
    horizon,
    fold,
    metric,
    value,
    n_test_rows,
    train_rows,
    -- The honest sample size: daily rows at horizon h overlap h-fold, so 4,305 rows at
    -- h=365 are about 11.8 independent observations, not 4,305. Carried on every row so
    -- a long-horizon score cannot be read as if it were a short-horizon one.
    non_overlapping_windows,
    reason,
    panel_vintage_ts
from read_parquet('{{ var("data_root") }}/gold/model_score/*.parquet')
