-- Current-vintage view of the bi-temporal silver panel.
-- Point-in-time analysis in Plan 4 reads the panel directly with an
-- as-of filter instead of this model.
select
    asset,
    cast(metric_date as date)      as metric_date,
    "CapMrktCurUSD"                as market_cap_usd,
    "CapMVRVCur"                   as mvrv,
    realized_cap,
    "PriceUSD"                     as price_usd,
    "SplyCur"                      as supply,
    "HashRate"                     as hash_rate,
    "IssTotUSD"                    as issuance_usd,
    "FeeTotNtv"                    as fees_native,
    "AdrActCnt"                    as active_addresses,
    "TxCnt"                        as tx_count,
    effective_start_ts,
    effective_end_ts,
    row_hash
from read_parquet('{{ var("data_root") }}/silver/asset_metrics/*.parquet')
where is_current
