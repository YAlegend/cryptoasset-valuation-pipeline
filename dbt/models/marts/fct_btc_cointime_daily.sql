-- Family E(b)'s inputs at grain (symbol, metric_date): Cointime primitives, HODL waves by
-- supply and by realized cap, short-term-holder cohort and realized profit. BTC only.
select * from {{ ref('stg_btc_cointime') }}
