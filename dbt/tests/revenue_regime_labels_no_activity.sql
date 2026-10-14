-- M1: a day with zero fees AND zero revenue fell through to 'retains_revenue' on 252 rows,
-- which reads as a protocol choosing to keep nothing from nothing. Nothing happened that
-- day, and the label has to say so.
select symbol, metric_date, total_fees_usd, net_revenue_usd, revenue_regime
from {{ ref('fct_protocol_revenue_daily') }}
where coalesce(total_fees_usd, 0) = 0
  and net_revenue_usd = 0
  and revenue_regime != 'no_activity'
