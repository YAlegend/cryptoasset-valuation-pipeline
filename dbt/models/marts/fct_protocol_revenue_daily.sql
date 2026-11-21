-- The history series is the ONLY spine. The snapshot used to supply any (symbol, date) the
-- history lacked, which manufactured a duplicate day: the snapshot stamps DefiLlama's
-- rolling total24h with the capture date, and for a protocol whose history lags that value
-- IS the previous day's complete figure (LDO 2026-10-14 was LDO 2026-10-13 to the dollar).
-- A trailing-window consumer then counted one day of cash flow twice, which is the exact
-- failure this mart exists to prevent, so the snapshot now contributes one column and no
-- rows (left join, so a missing snapshot cannot drop a history row either).
--
-- total_fees_30d is a 30-day aggregate the history endpoint does not serve, so it arrives
-- from the snapshot where the snapshot covers that (symbol, date) and is NULL elsewhere.
with history as (
    select
        symbol,
        metric_date,
        total_fees_usd,
        net_revenue_usd,
        protocol_count,
        methodology_hash
    from {{ ref('stg_protocol_fee_history') }}
),
snapshot as (
    select symbol, metric_date, total_fees_30d
    from {{ ref('stg_protocol_revenue') }}
),
composed as (
    select
        h.symbol,
        h.metric_date,
        h.total_fees_usd,
        s.total_fees_30d,
        h.net_revenue_usd,
        h.protocol_count,
        -- The components that reported on the preceding day. A tail row built from fewer
        -- of them is a composition artefact, not a collapse in cash flow: MKR read 6,708
        -- against 975,064 the day before because one adapter had not published yet.
        lag(h.protocol_count) over (partition by h.symbol order by h.metric_date)
            as protocol_count_prior_day,
        h.methodology_hash,
        'history' as series_source
    from history h
    left join snapshot s
      on h.symbol = s.symbol and h.metric_date = s.metric_date
)
select
    *,
    -- False marks a day that lost a component. A genuine retirement keeps it false from
    -- then on, which is visible and conservative; a launch raises the count and keeps it
    -- true. A consumer averaging a window must require this (Task 10).
    coalesce(protocol_count >= protocol_count_prior_day, true) as composition_complete,
    {{ revenue_regime() }} as revenue_regime
from composed
