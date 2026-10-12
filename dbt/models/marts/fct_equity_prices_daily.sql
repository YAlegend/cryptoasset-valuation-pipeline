-- Equities and macro series. Separate from the crypto fact by design (spec 8.1): a
-- different trading calendar means a shared date spine would manufacture nulls on every
-- weekend and holiday.
select
    e.symbol,
    e.metric_date,
    e.price_usd,
    a.sector
from {{ ref('stg_equity_prices') }} e
left join {{ ref('assets') }} a on e.symbol = a.symbol
