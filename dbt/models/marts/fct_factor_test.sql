-- Spec 11.5 at grain (model_id, signal, portfolio, metric), on the Coin Metrics community
-- universe (design 2026-10-21-factor-models). Weekly, full sample, descriptive: the signals
-- and floors were chosen in 2026 with the literature in hand, so no tradability claim.
--
-- portfolio is D1..D10 or HL on decile and alpha rows, NULL otherwise. model_id = '*' holds
-- the hypothesis count: the family is every 'p_value' row (13 when nothing is refused).
-- model_id = 'panel', metric = 'spikes_nulled' counts Sunday prices nulled as vendor spikes.
select
    model_id,
    signal,
    portfolio,
    metric,
    value,
    n_weeks,
    reason,
    params_hash,
    panel_vintage_ts
from read_parquet('{{ var("data_root") }}/gold/factor_test/*.parquet')
