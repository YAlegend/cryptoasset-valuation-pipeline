-- The conformed daily fact table: one row per asset per day, with the
-- derived cost-approach quantities the spec requires (section 6.2).
with base as (
    select * from {{ ref('stg_asset_metrics') }}
),
derived as (
    select
        *,
        issuance_usd + (fees_native * price_usd)                    as miner_revenue_usd,
        sum(issuance_usd + (fees_native * price_usd)) over (
            partition by asset order by metric_date
            rows between unbounded preceding and current row
        )                                                           as thermocap_usd
    from base
)
select
    *,
    case when thermocap_usd > 0 then market_cap_usd / thermocap_usd end as mvtv,
    realized_cap - thermocap_usd                                        as investor_cap_usd
from derived
