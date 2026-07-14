"""Heliostream data-engineering layer.

A streaming + batch pipeline that lands upstream solar-wind measurements in a
DuckDB warehouse, transforms them with dbt into a tested feature mart, and feeds
the Heliostream model from the warehouse instead of an in-memory frame.

Message bus:  Kafka/Redpanda (real) or a file-backed log (offline/testing).
Warehouse:    DuckDB (raw -> staging -> features), transformed by dbt with tests.
"""
__version__ = "0.1.0"
