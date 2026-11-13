-- The weekly series behind fct_factor_test: HL_<signal> decile spreads and the LTW
-- factors CMKT, CSMB, CMOM, each the week-t sort's return over week t -> t+1.
select
    week,
    series,
    value
from read_parquet('{{ var("data_root") }}/gold/factor_returns/*.parquet')
