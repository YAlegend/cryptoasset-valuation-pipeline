-- Warn, not error: the vendor's two series are computed independently, so net revenue above
-- gross fees is a vendor-semantics disagreement rather than a defect of ours. It is worth
-- seeing, because a DCF reading it as holder cash flow is reading more than the fees it came
-- from. 18 LINK rows breach it today (Chainlink Staking, by up to 76,669).
{{ config(severity = 'warn') }}
select symbol, metric_date, total_fees_usd, net_revenue_usd
from {{ ref('fct_protocol_revenue_daily') }}
where net_revenue_usd > total_fees_usd
