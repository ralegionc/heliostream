-- Cleaned, deduplicated solar-wind hours. One row per timestamp (latest land).
with src as (
    select * from {{ source('raw', 'solar_wind') }}
)
select
    cast(time as timestamp)        as time,
    bt, bz, by_gsm, v, n, dst,
    source,
    ingested_at
from src
where v  is not null and v between 100 and 3000   -- drop obviously bad speed
  and bt is not null and bt >= 0
qualify row_number() over (partition by time order by ingested_at desc) = 1
