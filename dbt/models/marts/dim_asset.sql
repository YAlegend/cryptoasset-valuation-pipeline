-- Asset dimension derived from what the panel actually contains.
-- Plan 2 joins sector and consensus from config/assets.yml as a dbt seed;
-- this model establishes the grain and the observed date coverage.
select
    asset,
    min(metric_date) as first_observed_on,
    max(metric_date) as last_observed_on,
    count(*)         as observation_days
from {{ ref('stg_asset_metrics') }}
group by asset
