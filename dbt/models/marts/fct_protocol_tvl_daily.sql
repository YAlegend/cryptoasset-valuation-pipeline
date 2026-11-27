-- TVL at the valuation grain. protocol_count travels with the value for the reason
-- fct_protocol_revenue_daily gives: a day that lost a component reads lower for a
-- composition reason, not an economic one, and P/TVL would call that a cheapening.
select
    symbol,
    metric_date,
    tvl_usd,
    protocol_count,
    lag(protocol_count) over (partition by symbol order by metric_date)
        as protocol_count_prior_day,
    coalesce(
        protocol_count >= lag(protocol_count) over (partition by symbol order by metric_date),
        true
    ) as composition_complete
from {{ ref('stg_protocol_tvl') }}
