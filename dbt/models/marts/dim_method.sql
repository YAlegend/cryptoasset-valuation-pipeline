-- Seeded from config/methods/, which makes "which methods disagree, and why" a SQL
-- question and is the source of the method cards (spec 8.1).
select
    method_id,
    family,
    requires,
    min_density,
    min_rows,
    citation,
    returns_fair_value
from {{ ref('methods') }}
