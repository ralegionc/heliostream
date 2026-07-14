-- Physical invariant: the GSM Bz component cannot exceed the field magnitude.
-- A violation means corrupt or mis-joined data. Test passes iff zero rows.
select time, bz, bt
from {{ ref('features_hourly') }}
where abs(bz) > bt + 0.01
