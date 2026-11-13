-- Equities and macro series, kept separate from the crypto panel: a different trading
-- calendar, a different freshness SLA (144h, because markets close), and no network
-- metrics at all (spec 8.1).
select
    a.symbol,
    cast(e.metric_date at time zone 'UTC' as date) as metric_date,
    e.price_usd
from read_parquet('{{ var("data_root") }}/silver/equity_prices/*.parquet') e
join {{ ref('assets') }} a on e.asset = a.yahoo_id
where e.is_current
