-- Protocol revenue at the valuation grain. Only mapped assets appear: the feed carries
-- 2,991 protocols and at most a handful map to anything we value (spec 5.1).
select
    symbol,
    metric_date,
    total_fees_24h,
    total_fees_30d,
    net_revenue_24h,
    protocol_count,
    methodology_hash,
    -- The denominator discipline spec 10.4 demands, made explicit: a zero net revenue
    -- against non-zero fees is the Uniswap case -- the protocol fee switch is off and all
    -- fees go to liquidity providers -- which is a finding, not a gap.
    case
        when net_revenue_24h is null then 'unknown'
        when net_revenue_24h = 0 and total_fees_24h > 0 then 'fees_fully_distributed'
        when net_revenue_24h < 0 then 'net_outflow'
        else 'retains_revenue'
    end as revenue_regime
from {{ ref('stg_protocol_revenue') }}
