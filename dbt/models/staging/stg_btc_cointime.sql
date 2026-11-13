-- Bitcoin coin-age series from the BigQuery UTXO extraction (Plan 3b). Empty until the
-- backfill from genesis is complete: Cointime is cumulative, so it is all-or-nothing (spec 6.3).
select
    'BTC' as symbol,
    cast(metric_date as date) as metric_date,
    * exclude (metric_date)
from read_parquet('{{ var("data_root") }}/silver/btc_cointime_daily/*.parquet')
