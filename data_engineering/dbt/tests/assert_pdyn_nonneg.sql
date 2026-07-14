-- Dynamic pressure is defined non-negative. Passes iff zero rows.
select time, pdyn
from {{ ref('features_hourly') }}
where pdyn < 0
