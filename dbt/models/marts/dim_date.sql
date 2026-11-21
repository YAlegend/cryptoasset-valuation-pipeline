-- Date dimension carrying Bitcoin halving context, so cycle metrics in
-- Plan 2 are a join rather than a recalculation (spec section 8.1).
with bounds as (
    select min(metric_date) as d0, max(metric_date) as d1 from {{ ref('stg_asset_metrics') }}
),
spine as (
    select unnest(generate_series(
        (select d0 from bounds),
        (select d1 from bounds),
        interval 1 day
    ))::date as date_day
),
halvings as (
    select * from (values
        (0, date '2009-01-03'),
        (1, date '2012-11-28'),
        (2, date '2016-07-09'),
        (3, date '2020-05-11'),
        (4, date '2024-04-19'),
        (5, date '2028-04-01')   -- projected; refreshed when the real date is known
    ) as t(epoch, started_on)
)
select
    s.date_day,
    extract(year from s.date_day)                      as year,
    extract(month from s.date_day)                     as month,
    s.date_day = (date_trunc('month', s.date_day) + interval 1 month - interval 1 day)::date
                                                       as is_month_end,
    h.epoch                                            as halving_epoch,
    date_diff('day', h.started_on, s.date_day)         as days_since_halving
from spine s
left join halvings h
    on h.started_on <= s.date_day
   and s.date_day < coalesce(
        (select min(h2.started_on) from halvings h2 where h2.started_on > h.started_on),
        date '9999-12-31'
   )
