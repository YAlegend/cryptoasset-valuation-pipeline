-- The conformed daily fact: one row per (symbol, date), joining Coin Metrics' network
-- metrics, its valuation metrics where free, and CoinGecko's panel.
--
-- Two deliberate asymmetries (spec 10.1):
--  * VOLUME is never coalesced. Coin Metrics' reported spot volume and CoinGecko's total
--    volume are different constructions with different exchange coverage -- 2.97e9 against
--    1.48e10 for BTC on the same day. Velocity, NVT and P/U all turn on this term, so both
--    are carried and every method names which it used.
--  * MARKET CAP is resolved, because every method needs one, but market_cap_source records
--    where it came from: TRX has no free Coin Metrics market cap at all (spec 5.1).
--
-- A NULL input day contributes 0 to thermocap (so the series stays usable as a lower
-- bound) and raises thermocap_has_gap from that row onward.
with base as (
    select
        m.*,
        c.cg_price_usd,
        c.cg_market_cap_usd,
        c.cg_volume_total_usd,
        v.source_kind as valuation_source_kind
    from {{ ref('stg_asset_metrics') }} m
    left join {{ ref('stg_market_chart') }} c
           on m.symbol = c.symbol and m.metric_date = c.metric_date
    left join (
        select symbol, source_kind
        from {{ ref('asset_metric_sources') }}
        where metric_group = 'valuation'
    ) v on m.symbol = v.symbol
),
resolved as (
    select
        *,
        case
            when valuation_source_kind = 'coingecko' then cg_market_cap_usd
            else market_cap_usd
        end as resolved_market_cap_usd,
        -- NULL exactly when the resolved market cap is NULL: the source is only named
        -- where there is a figure to attribute.
        case
            when valuation_source_kind = 'coingecko' then
                case when cg_market_cap_usd is not null then 'coingecko' end
            when market_cap_usd is not null then 'coinmetrics'
        end as market_cap_source
    from base
),
derived as (
    select
        *,
        issuance_usd + (fees_native * price_usd)                    as miner_revenue_usd,
        sum(coalesce(issuance_usd + (fees_native * price_usd), 0)) over (
            partition by symbol order by metric_date
            rows between unbounded preceding and current row
        )                                                           as thermocap_usd,
        coalesce(bool_or(issuance_usd is null or fees_native is null or price_usd is null) over (
            partition by symbol order by metric_date
            rows between unbounded preceding and current row
        ), false)                                                   as thermocap_has_gap
    from resolved
)
select
    symbol,
    metric_date,
    resolved_market_cap_usd                        as market_cap_usd,
    market_cap_source,
    mvrv,
    realized_cap,
    coalesce(price_usd, cg_price_usd)              as price_usd,
    supply,
    hash_rate,
    issuance_usd,
    fees_native,
    active_addresses,
    tx_count,
    volume_reported_spot_usd,
    cg_volume_total_usd                            as volume_total_usd,
    -- Two velocities, one per volume definition. Zero or negative denominators yield NULL,
    -- never infinity.
    case when resolved_market_cap_usd > 0
         then volume_reported_spot_usd / resolved_market_cap_usd end as velocity_reported,
    case when resolved_market_cap_usd > 0
         then cg_volume_total_usd / resolved_market_cap_usd end      as velocity_total,
    miner_revenue_usd,
    thermocap_usd,
    thermocap_has_gap,
    case when thermocap_usd > 0
         then resolved_market_cap_usd / thermocap_usd end            as mvtv,
    realized_cap - thermocap_usd                                     as investor_cap_usd,
    -- The panel vintage this row was built from. Review Focus 4: a quarantined cell
    -- changes velocity and therefore every Family A and D estimate, so an estimate must
    -- be traceable to the panel state that produced it. effective_end_ts is deliberately
    -- absent: this is a current-state grain built downstream of an is_current filter, where
    -- it would always be the open sentinel. Bi-temporal reconstruction belongs in silver.
    effective_start_ts                                               as panel_vintage_ts
from derived
