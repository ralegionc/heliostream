-- Hourly feature mart consumed by the Heliostream model.
-- Cleaning note: a field component's magnitude cannot physically exceed the
-- total field magnitude, so |Bz| is clamped to Bt. Real OMNI reports the field
-- magnitude and the Bz component from independent, separately-rounded fields,
-- so a handful of low-field hours violate this at the rounding level; clamping
-- restores physical consistency for the coupling and clock-angle features.
with stg as (
    select * from {{ ref('stg_solar_wind') }}
),
clean as (
    select
        time,
        bt,
        greatest(-bt, least(bt, bz)) as bz,   -- clamp |Bz| <= Bt
        by_gsm, v, n, dst, source
    from stg
)
select
    time,
    bt, bz, by_gsm, v, n, dst,
    source,
    greatest(0.0, -bz)                       as bs,
    v * greatest(0.0, -bz) * 0.001           as vbs,
    1.6726e-6 * n * v * v                     as pdyn,
    pow(sin(atan2(by_gsm, bz) / 2.0), 4)      as sin_clock_half4
from clean
order by time