-- CoinGecko's current supply figures resolved to `symbol`. A snapshot, not a series: the
-- free tier has no supply history. max_supply is NULL for an asset with no cap (ETH), and
-- stays NULL -- it is never filled from total_supply (an FDV multiple would become a
-- market-cap multiple under an FDV label).
--
-- supply_as_of is the date the CURRENT values were first observed (the SCD2 version's
-- effective_start_ts), not the date of the latest run: the table stores no as-of column
-- because one would be hashed into row_hash and re-version every asset on every daily
-- ingest. Unchanged values keep their old as-of; a restatement moves it.
select
    a.symbol,
    s.circulating_supply,
    s.total_supply,
    s.max_supply,
    s.fully_diluted_valuation as vendor_fully_diluted_valuation,
    cast(s.effective_start_ts at time zone 'UTC' as date) as supply_as_of
from read_parquet('{{ var("data_root") }}/silver/supply_metadata/*.parquet') s
join {{ ref('assets') }} a on a.gecko_id = s.gecko_id
where s.is_current
