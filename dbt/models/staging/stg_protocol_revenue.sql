-- DefiLlama rolled up to `symbol`. A protocol's revenue is split across versions and
-- renamed on rebrand, so the mapping is a LIST of slugs (spec 5.1): UNI is Uniswap V4
-- plus V3 plus V2, and MKR is Sky Lending plus Sky Money.
--
-- sum(...) over an all-null group would return 0.0 in SQL, which asserts "retained
-- nothing" rather than "unknown", so each column is nulled back out when no input row
-- carried a value.
with mapped as (
    select
        a.symbol,
        cast(r.metric_date at time zone 'UTC' as date) as metric_date,
        r.protocol,
        r.total_fees_24h,
        r.total_fees_30d,
        r.net_revenue_24h,
        r.methodology_hash
    from read_parquet('{{ var("data_root") }}/silver/protocol_revenue/*.parquet') r
    join {{ ref('assets') }} a
      on list_contains(string_split(a.defillama_slugs, '|'), r.protocol)
    where r.is_current
)
select
    symbol,
    metric_date,
    case when count(total_fees_24h)  > 0 then sum(total_fees_24h)  end as total_fees_24h,
    case when count(total_fees_30d)  > 0 then sum(total_fees_30d)  end as total_fees_30d,
    case when count(net_revenue_24h) > 0 then sum(net_revenue_24h) end as net_revenue_24h,
    count(*)                                                           as protocol_count,
    -- One hash per symbol-day: a change in any component protocol's methodology
    -- invalidates the rolled-up series (spec 9.2).
    md5(string_agg(coalesce(methodology_hash, ''), '|' order by protocol))
                                                                       as methodology_hash
from mapped
group by symbol, metric_date
