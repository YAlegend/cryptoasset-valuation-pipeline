{#- The denominator discipline spec 10.4 demands, made explicit, in one place so the history
    branch and the snapshot branch of fct_protocol_revenue_daily classify identically. A zero
    net revenue against non-zero fees is the Uniswap case -- the protocol fee switch is off
    and all fees go to liquidity providers -- which is a finding, not a gap. -#}
{% macro revenue_regime() %}
    case
        when net_revenue_usd is null then 'unknown'
        when net_revenue_usd = 0 and total_fees_usd > 0 then 'fees_fully_distributed'
        when net_revenue_usd < 0 then 'net_outflow'
        else 'retains_revenue'
    end
{% endmacro %}
