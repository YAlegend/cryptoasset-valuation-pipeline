-- DefiLlama's per-protocol daily history rolled up to `symbol` across the configured slug
-- list, by the same rule as stg_protocol_revenue (UNI = V4 + V3 + V2; MKR = Sky Lending +
-- Sky Money). Each column is nulled back out when no input row carried a value, because
-- sum(...) over an all-null group would return 0.0 and assert "retained nothing".
with mapped as (
    select
        a.symbol,
        cast(r.metric_date at time zone 'UTC' as date) as metric_date,
        r.protocol,
        r.total_fees_usd,
        r.net_revenue_usd,
        r.methodology_hash
    from read_parquet('{{ var("data_root") }}/silver/protocol_fee_history/*.parquet') r
    join {{ ref('assets') }} a
      on list_contains(string_split(a.defillama_slugs, '|'), r.protocol)
    where r.is_current
)
select
    symbol,
    metric_date,
    case when count(total_fees_usd)  > 0 then sum(total_fees_usd)  end as total_fees_usd,
    case when count(net_revenue_usd) > 0 then sum(net_revenue_usd) end as net_revenue_usd,
    count(*)                                                           as protocol_count,
    md5(string_agg(coalesce(methodology_hash, ''), '|' order by protocol))
                                                                       as methodology_hash
from mapped
group by symbol, metric_date
