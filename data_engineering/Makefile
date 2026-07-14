# Heliostream data-engineering pipeline.
# Offline targets need no broker/Docker; docker-* targets use Redpanda.

export PYTHONPATH := .
export HELIO_DUCKDB := $(CURDIR)/warehouse/heliostream.duckdb

.PHONY: install demo batch produce consume dbt quality train stats clean \
        docker-up docker-dbt docker-down test

install:            ## install pipeline + model package (editable)
	pip install -r requirements.txt && pip install -e ..

demo: batch train   ## full offline demo: DAG then train off the warehouse

batch:              ## run the offline DAG: produce -> load -> dbt build+test -> quality
	python -m heliostream_de run-batch --source synthetic --hours 26280

produce:            ## synthetic -> bus (file backend)
	python -m heliostream_de produce --source synthetic --hours 26280

consume:            ## bus -> raw.solar_wind
	python -m heliostream_de consume

dbt:                ## transform + test (staging, feature mart)
	python -m heliostream_de dbt build

quality:            ## warehouse data-quality gate
	python -m heliostream_de quality

train:              ## train the model directly off the feature mart
	python -m heliostream_de train --model hybrid --epochs 40

stats:              ## raw landing stats
	python -m heliostream_de stats

test:               ## pipeline unit tests
	python -m pytest tests/ -q

clean:              ## wipe the local warehouse + bus
	rm -rf warehouse

docker-up:          ## start Redpanda + console + streaming producer/consumer
	docker compose up -d redpanda console consumer producer

docker-dbt:         ## run dbt inside the stack
	docker compose run --rm dbt build

docker-down:        ## stop the stack
	docker compose down -v
