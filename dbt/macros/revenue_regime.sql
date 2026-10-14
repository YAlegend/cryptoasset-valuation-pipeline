{#- The denominator discipline spec 10.4 demands, made explicit, in one place. A zero net
    revenue against non-zero fees is the Uniswap case -- the fee switch is off and all fees
    go to liquidity providers -- which is a finding, not a gap. Zero fees AND zero revenue
    is neither: nothing happened that day, and calling that "retains_revenue" (252 rows did)
    reads as a protocol choosing to keep nothing from nothing. -#}
{% macro revenue_regime() %}
    case
        when net_revenue_usd is null then 'unknown'
        when coalesce(total_fees_usd, 0) = 0 and net_revenue_usd = 0 then 'no_activity'
        when net_revenue_usd = 0 and total_fees_usd > 0 then 'fees_fully_distributed'
        when net_revenue_usd < 0 then 'net_outflow'
        else 'retains_revenue'
    end
{% endmacro %}
