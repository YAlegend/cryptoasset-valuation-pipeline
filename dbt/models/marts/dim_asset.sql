-- Seeded from config/assets.yml, not derived from the panel (spec 8.1). Observed coverage
-- is joined on as an attribute, so "what does this asset actually have" stays answerable
-- without making it the dimension's grain.
with observed as (
    select
        symbol,
        min(metric_date) as first_observed_on,
        max(metric_date) as last_observed_on,
        count(*)         as observation_days,
        -- Density per load-bearing column: a method's coverage floor is checked against
        -- these (spec 10.1, Review Focus 3 -- BNB's market cap is 19% dense).
        count(market_cap_usd)::double / nullif(count(*), 0) as market_cap_density,
        count(active_addresses)::double / nullif(count(*), 0) as active_address_density,
        count(volume_reported_spot_usd)::double / nullif(count(*), 0) as volume_density,
        count(hash_rate)::double / nullif(count(*), 0) as hash_rate_density
    from {{ ref('stg_asset_metrics') }}
    group by symbol
)
select
    a.symbol,
    a.sector,
    a.consensus,
    a.is_stablecoin,
    a.is_equity,
    a.cm_id,
    a.gecko_id,
    a.yahoo_id,
    a.defillama_slugs,
    a.coverage_note,
    o.first_observed_on,
    o.last_observed_on,
    coalesce(o.observation_days, 0) as observation_days,
    o.market_cap_density,
    o.active_address_density,
    o.volume_density,
    o.hash_rate_density
from {{ ref('assets') }} a
left join observed o on a.symbol = o.symbol
