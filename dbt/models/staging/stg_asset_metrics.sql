-- Current-vintage view of the bi-temporal silver panel, resolved from Coin Metrics'
-- vendor ids to the conformed `symbol` key (spec 5.1).
--
-- Three assets draw different metric groups from different vendor series: AVAX takes
-- volume from `avax` because `avaxc` serves none, and TRX takes valuation from CoinGecko
-- because `cm:trx` has no free market cap. So this is one join per group, not one per
-- asset, and the groups are recombined on (symbol, metric_date).
--
-- Point-in-time analysis in Plan 4 reads the panel directly with an as-of filter.
with panel as (
    select
        asset as vendor_id,
        -- Convert to UTC before truncating: a bare cast uses the session TimeZone and
        -- shifts the date by a day on non-UTC machines.
        cast(metric_date at time zone 'UTC' as date) as metric_date,
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
        "volume_reported_spot_usd_1d"  as volume_reported_spot_usd,
        effective_start_ts,
        effective_end_ts,
        row_hash
    from read_parquet('{{ var("data_root") }}/silver/asset_metrics/*.parquet')
    where is_current
),
sources as (
    select symbol, metric_group, source_id
    from {{ ref('asset_metric_sources') }}
    where source_kind = 'cm'
),
network as (
    select s.symbol, p.metric_date, p.supply, p.hash_rate, p.issuance_usd, p.fees_native,
           p.active_addresses, p.tx_count, p.effective_start_ts, p.row_hash
    from panel p
    join sources s on p.vendor_id = s.source_id and s.metric_group = 'network'
),
valuation as (
    select s.symbol, p.metric_date, p.market_cap_usd, p.mvrv, p.realized_cap, p.price_usd
    from panel p
    join sources s on p.vendor_id = s.source_id and s.metric_group = 'valuation'
),
volume as (
    select s.symbol, p.metric_date, p.volume_reported_spot_usd
    from panel p
    join sources s on p.vendor_id = s.source_id and s.metric_group = 'volume'
),
spine as (
    select symbol, metric_date from network
    union
    select symbol, metric_date from valuation
    union
    select symbol, metric_date from volume
)
select
    sp.symbol,
    sp.metric_date,
    v.market_cap_usd,
    v.mvrv,
    v.realized_cap,
    v.price_usd,
    n.supply,
    n.hash_rate,
    n.issuance_usd,
    n.fees_native,
    n.active_addresses,
    n.tx_count,
    vol.volume_reported_spot_usd,
    n.effective_start_ts,
    n.row_hash
from spine sp
left join network   n   on sp.symbol = n.symbol   and sp.metric_date = n.metric_date
left join valuation v   on sp.symbol = v.symbol   and sp.metric_date = v.metric_date
left join volume    vol on sp.symbol = vol.symbol and sp.metric_date = vol.metric_date
