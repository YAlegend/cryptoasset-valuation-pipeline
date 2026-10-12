-- CoinGecko's panel, resolved to `symbol`. This is the only market cap available for TRX
-- and SOL, and its volume is a DIFFERENT series from Coin Metrics' reported spot volume
-- (2.97e9 vs 1.48e10 for BTC on the same day) -- never interchange them (spec 10.1).
select
    a.symbol,
    cast(m.metric_date at time zone 'UTC' as date) as metric_date,
    m.price_usd        as cg_price_usd,
    m.market_cap_usd   as cg_market_cap_usd,
    m.volume_usd       as cg_volume_total_usd,
    m.velocity         as cg_velocity,
    m.holding_time_days as cg_holding_time_days
from read_parquet('{{ var("data_root") }}/silver/market_chart/*.parquet') m
join {{ ref('assets') }} a on m.asset = a.gecko_id
where m.is_current
