-- Method registry output (spec 8.1). param_set_hash lets a DCF at 12% and at 16% coexist
-- as rows rather than overwrite, and panel_vintage_ts ties an estimate to the panel state
-- it was computed from.
--
-- Inapplicable rows are kept deliberately: "this method cannot be applied to this asset,
-- and here is the measurement that says so" is a research finding.
select
    symbol,
    metric_date,
    method_id,
    param_set_hash,
    fair_value,
    band_low,
    band_high,
    applicable,
    reason,
    circular_inputs,
    nullif(circular_inputs, '') is not null as is_circular,
    inputs_used,
    diagnostics,
    panel_vintage_ts
from read_parquet('{{ var("data_root") }}/gold/valuation_estimate/*.parquet')
