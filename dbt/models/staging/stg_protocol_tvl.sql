-- DefiLlama's per-protocol daily TVL rolled up to `symbol` across the configured slug
-- list, by the same rule as stg_protocol_fee_history (UNI = V4 + V3 + V2; MKR = Sky
-- Lending + Sky Money). Summing TVL across a symbol's own protocols is addition of
-- balances, not of flows, so no double-count arises from the rollup itself -- but it DOES
-- arise between symbols (ETH's chain TVL contains the Aave pools counted under AAVE), so
-- this grain answers "P/TVL for this protocol family" and nothing wider.
with tvl as (
    select
        a.symbol,
        cast(t.metric_date at time zone 'UTC' as date) as metric_date,
        t.protocol,
        t.tvl_usd
    from read_parquet('{{ var("data_root") }}/silver/protocol_tvl/*.parquet') t
    join {{ ref('assets') }} a
      on list_contains(string_split(a.defillama_slugs, '|'), t.protocol)
    where t.is_current
      -- Also here, not only at parse time: merge_scd2 does not close a key that vanishes
      -- from a later fetch, so a partial row captured mid-day on a previous run stays
      -- `is_current` forever once the parser stops sending it (Ruling 37).
      and cast(t.metric_date at time zone 'UTC' as date)
          < cast(current_timestamp at time zone 'UTC' as date)
)
select
    symbol,
    metric_date,
    -- Nulled back out when no component reported: sum(...) over an all-null group returns
    -- 0.0, which would assert "nothing is locked" where the truth is "we did not measure".
    case when count(tvl_usd) > 0 then sum(tvl_usd) end as tvl_usd,
    count(*)                                          as protocol_count
from tvl
group by symbol, metric_date
