-- The history series is the spine; the snapshot supplies only the (symbol, date) pairs the
-- history does not have. series_source names which produced each row, because a DCF built
-- half on one and half on the other is not reproducible.
--
-- total_fees_30d exists only on snapshot rows: the history endpoint serves a daily series,
-- not a 30-day aggregate, so it is NULL on every history row rather than computed here.
with history as (
    select
        symbol,
        metric_date,
        total_fees_usd,
        cast(null as double) as total_fees_30d,
        net_revenue_usd,
        protocol_count,
        methodology_hash,
        'history' as series_source
    from {{ ref('stg_protocol_fee_history') }}
),
snapshot as (
    select
        symbol,
        metric_date,
        total_fees_24h as total_fees_usd,
        total_fees_30d,
        net_revenue_24h as net_revenue_usd,
        protocol_count,
        methodology_hash,
        'snapshot' as series_source
    from {{ ref('stg_protocol_revenue') }}
    where (symbol, metric_date) not in (select symbol, metric_date from history)
),
unioned as (
    select * from history
    union all
    select * from snapshot
)
select *, {{ revenue_regime() }} as revenue_regime
from unioned
