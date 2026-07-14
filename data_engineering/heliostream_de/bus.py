"""Message bus with two interchangeable backends.

KafkaBus  - real Kafka/Redpanda (used with docker-compose).
FileBus   - an append-only JSON-lines log on disk that emulates a topic with a
            durable consumer offset, so the full producer -> bus -> consumer path
            runs and is testable with no broker.

Both expose: produce(record) / flush() and consume(max_records, timeout).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import config as C


class FileBus:
    """Append-only log file acting as a single-partition topic."""

    def __init__(self, path=None, group=None):
        self.path = Path(path or C.FILE_BUS_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._offset_file = self.path.with_suffix(f".{group or C.CONSUMER_GROUP}.offset")

    # --- producer ---
    def produce(self, record: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def flush(self):
        pass

    # --- consumer ---
    def _read_offset(self) -> int:
        if self._offset_file.exists():
            return int(self._offset_file.read_text() or "0")
        return 0

    def _write_offset(self, n: int):
        self._offset_file.write_text(str(n))

    def consume(self, max_records=10_000, timeout=0.0):
        """Return up to max_records new records since the committed offset."""
        start = self._read_offset()
        out, line_no = [], 0
        with open(self.path) as f:
            for line_no, line in enumerate(f):
                if line_no < start:
                    continue
                line = line.strip()
                if line:
                    out.append(json.loads(line))
                if len(out) >= max_records:
                    line_no += 1
                    break
            else:
                line_no = line_no + 1 if (line_no or out) else start
        self._write_offset(max(start, line_no))
        return out


class KafkaBus:
    """Kafka/Redpanda backend via kafka-python (JSON values, `time` as key)."""

    def __init__(self, brokers=None, topic=None, group=None):
        from kafka import KafkaProducer, KafkaConsumer  # lazy: only when used
        self.topic = topic or C.TOPIC_RAW
        self._KafkaConsumer = KafkaConsumer
        self.brokers = (brokers or C.KAFKA_BROKERS).split(",")
        self.group = group or C.CONSUMER_GROUP
        self._producer = KafkaProducer(
            bootstrap_servers=self.brokers,
            value_serializer=lambda v: json.dumps(v, default=str).encode(),
            key_serializer=lambda k: str(k).encode(),
            acks="all", linger_ms=200,
        )
        self._consumer = None

    def produce(self, record: dict):
        self._producer.send(self.topic, key=record.get("time"), value=record)

    def flush(self):
        self._producer.flush()

    def _ensure_consumer(self):
        if self._consumer is None:
            self._consumer = self._KafkaConsumer(
                self.topic, bootstrap_servers=self.brokers, group_id=self.group,
                enable_auto_commit=True, auto_offset_reset="earliest",
                value_deserializer=lambda b: json.loads(b.decode()),
                consumer_timeout_ms=2000,
            )

    def consume(self, max_records=10_000, timeout=2.0):
        self._ensure_consumer()
        out = []
        deadline = time.time() + timeout
        for msg in self._consumer:
            out.append(msg.value)
            if len(out) >= max_records or time.time() > deadline:
                break
        return out


def get_bus(backend=None, **kw):
    backend = backend or C.BUS_BACKEND
    if backend == "kafka":
        return KafkaBus(**kw)
    return FileBus(**kw)
