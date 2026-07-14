# Heliostream

**Forecasting geomagnetic storms from the solar wind, with honest uncertainty.**

Heliostream watches the stream of charged particles blowing off the Sun and predicts,
one to six hours ahead, how badly Earth's magnetic field is about to be disturbed. It
does this with a model that has real physics built into it, it says how confident it is
rather than giving a single bare number, and it runs on top of a tested data pipeline
that catches bad data before the model ever sees it.

Everything here runs on a normal laptop. No GPU, no cloud account, no API keys.

---

## Table of contents

1. [What problem is this solving?](#1-what-problem-is-this-solving)
2. [The jargon, explained](#2-the-jargon-explained)
3. [What this project actually does](#3-what-this-project-actually-does)
4. [Results on real NASA data](#4-results-on-real-nasa-data)
5. [Quick start](#5-quick-start)
6. [The four things you can run](#6-the-four-things-you-can-run)
7. [How it works, layer by layer](#7-how-it-works-layer-by-layer)
8. [Two real bugs found in NASA's own data](#8-two-real-bugs-found-in-nasas-own-data)
9. [Repository map](#9-repository-map)
10. [Honest limitations](#10-honest-limitations)
11. [Credits and references](#11-credits-and-references)

---

## 1. What problem is this solving?

The Sun constantly blows a stream of electrically charged particles into space. When a
strong gust hits Earth, it shakes Earth's magnetic field. That shaking is called a
**geomagnetic storm**.

Storms are not just a light show. A strong one can:

- knock out electrical power grids (Quebec lost power for nine hours in 1989)
- scramble GPS accuracy, which matters for aviation, shipping, and farming
- damage satellites and shorten their lifespans
- disrupt radio communication, including for airlines on polar routes

There is a satellite sitting about 1.5 million kilometres from Earth, in the direction
of the Sun. It measures the solar wind *before* that wind reaches us. That gives roughly
an hour of advance warning.

**This project reads that upstream measurement and forecasts how bad the resulting storm
will be over the next few hours.** It is a weather forecast, except for space storms
instead of rain.

---

## 2. The jargon, explained

Every term this project uses, in plain language. You do not need any physics background.

### Space science terms

**Solar wind**
The continuous stream of charged particles (mostly protons and electrons) flowing out
from the Sun in all directions, at roughly 400 kilometres per second. It never stops. It
just gets faster and denser sometimes.

**Interplanetary magnetic field (IMF)**
The Sun's magnetic field, carried along by the solar wind. It has a direction. This turns
out to matter enormously (see **Bz** below).

**Bz**
The north-south component of that magnetic field. This is the single most important number
in the whole project. When Bz points **south** (a negative value), it is opposite to
Earth's own field, the two fields link up, and energy pours into Earth's magnetosphere.
Storms happen. When Bz points north, very little gets in. A fast solar wind with a
strongly southward Bz is the recipe for a big storm.

**Bt**
The total strength of the magnetic field, regardless of direction. Since Bz is one
component of the total field, |Bz| can never be larger than Bt. (This physical fact
becomes important in section 8.)

**By**
The east-west component of the field. Less important than Bz, but it affects the geometry
of how the fields connect.

**V**
The speed of the solar wind, in kilometres per second. Typically around 400 km/s, but can
exceed 800 km/s during fast streams.

**n**
The density of the solar wind, in particles per cubic centimetre. Typically around 5.

**Dst index (Disturbance storm time)**
**This is what the project predicts.** It is a single number, measured in nanotesla (nT),
that summarises how disturbed Earth's magnetic field is right now. It is computed from
magnetometers near the equator.

- Around 0 nT: quiet, nothing happening
- Below -50 nT: a storm is underway
- Below -100 nT: an intense storm
- Below -200 nT: severe, the kind that damages infrastructure
- The largest storm in our 10 years of data reached -234 nT (March 2015)

**More negative means a worse storm.** The reason it goes negative is that storms energise
a huge ring of electric current circling the Earth (the **ring current**), and that current
generates a magnetic field that *opposes* Earth's own, weakening the measurement.

**L1 (Lagrange point 1)**
A gravitationally convenient parking spot about 1.5 million km sunward of Earth, where
monitoring spacecraft (DSCOVR, ACE) sit. Solar wind measured at L1 reaches Earth roughly
an hour later. That hour is our warning time.

**OMNI**
A NASA dataset that merges measurements from several spacecraft into one clean, quality-
controlled record of the solar wind, hour by hour, going back to 1963. This is the project's
training data. It is "definitive," meaning carefully reprocessed after the fact.

**NOAA SWPC**
The US Space Weather Prediction Center, which publishes the solar wind *live*, right now,
as it arrives. This is what the live forecast reads.

**Nowcast**
A forecast for the very near future (hours, not days). Weather people use the word for
short-range prediction.

### Machine learning terms

**Model**
A program that learns patterns from examples. Here: given the last 24 hours of solar wind,
predict the Dst index for the next 1 to 6 hours.

**Training**
Showing the model thousands of historical examples so it can learn the pattern.

**Train / validation / test split**
The data is cut into three chronological pieces. The model *learns* on the first piece
(train), gets *tuned* on the second (validation), and is *judged* on the third (test),
which it has never seen. Splitting by time rather than randomly is essential here: shuffling
would let the model peek at the future, which would flatter the results dishonestly.

**RMSE (Root Mean Squared Error)**
The standard accuracy score. It is the typical size of the model's mistake, in the same
units as the thing being predicted (nT here). Lower is better. An RMSE of 8 nT means the
forecast is typically off by about 8 nT.

**Storm RMSE**
The same thing, but computed *only* on the hours when a storm was actually happening
(Dst < -50 nT). This matters because 97% of hours are quiet and boring. A model that just
predicts "everything is fine, always" scores a deceptively good overall RMSE while being
useless for the only hours anyone cares about. Storm RMSE is the honest number.

**Baseline**
A deliberately simple method used for comparison. If a fancy model cannot beat a simple
one, the fancy model is not earning its complexity. This project compares against three:

- **Persistence**: "whatever Dst is now, it will be the same in an hour." Surprisingly hard to beat.
- **O'Brien-McPherron**: a 25-year-old physics equation, no machine learning at all.
- **GRU**: a standard black-box neural network with no physics built in.

**GRU (Gated Recurrent Unit)**
A type of neural network designed for sequences (like a time series of solar wind readings).
It reads the last 24 hours and produces a summary. Used here both as the black-box baseline
and as one component inside the physics model.

**Uncertainty quantification**
Instead of predicting "Dst will be -80 nT," predict "Dst will be -80 nT, and I am 90%
confident it will land between -95 and -65." The range is often more useful than the number,
especially when deciding whether to act.

**Conformal prediction / calibration**
A statistical technique that adjusts the model's confidence ranges so they are *actually
honest*. If the model claims 90% confidence, the true value should really fall inside that
range 90% of the time. Models are usually overconfident by default. Conformal prediction
fixes this using held-out data, and comes with a mathematical guarantee. This project
achieves 88.5% coverage against a 90% target, which is close to ideal.

**Physics-informed model**
A model with known physical laws built into its structure, rather than one that has to
learn everything from scratch. See section 7 for how this one works. This is the project's
main technical contribution.

**Ordinary differential equation (ODE)**
An equation describing how something changes over time. The ring current has a well-known
one: energy gets injected while the solar wind drives it, and leaks away exponentially
afterwards. This project puts that equation *inside* the neural network.

### Data engineering terms

**Data pipeline**
The plumbing that moves data from where it lives (NASA's servers) to where the model needs
it, cleaning and checking it along the way.

**Ingestion**
Fetching the data and getting it into your own system.

**Message bus / Kafka**
A system that carries a stream of messages between programs. The part that fetches data
(**producer**) does not talk to the part that stores it (**consumer**) directly. Instead the
producer drops messages on the bus and the consumer picks them up. This means either side
can restart, crash, or slow down without losing data. **Kafka** and **Redpanda** are the
industry-standard implementations. This project can use Kafka, or a simple file-based
stand-in so it runs with no setup.

**Warehouse / DuckDB**
The database where cleaned data lives. **DuckDB** is a database that runs entirely inside a
single file with no server to install, which is why it works on a laptop.

**Raw / staging / mart (the three layers)**
A standard pattern:
- **Raw**: exactly what arrived, untouched. Never edited, so you can always go back.
- **Staging**: typed and deduplicated, but no business logic yet.
- **Mart**: the final, model-ready table with all the computed features.

**dbt (data build tool)**
The industry-standard tool for writing those transformations in SQL, with tests attached
and dependencies tracked automatically.

**Data quality test**
An automated check that fails loudly if the data is wrong. This project has 18, including
physics-based ones like "|Bz| must never exceed Bt." **These caught two real problems in
NASA's data** (section 8).

**Idempotent**
An operation you can safely run twice without doubling your data. If the same hour arrives
again, it overwrites rather than duplicates.

**Orchestration / Dagster**
The scheduler that runs the pipeline steps in the right order, on a schedule, and tells you
when something breaks. **Dagster** shows the whole thing as a picture: which step feeds which,
what ran when, what failed.

**Lineage**
The map of what data came from where. If a number looks wrong, lineage tells you every step
it passed through.

**Asset**
In Dagster, a thing that gets produced (a table, a trained model). You tell Dagster how to
build each asset and what it depends on, and it works out the order.

**Feature**
An input to the model. Some are measured directly (speed, density). Others are computed from
those, using physics. For example **VBs**, the product of speed and southward field, which
physically represents the electric field driving energy into Earth.

**Feature parity**
Proof that the features computed in SQL (in the warehouse) are *identical* to the ones
computed in Python (where the model was developed). Without this test, the two could silently
drift apart and the model would be fed subtly different inputs than it was validated on.

---

## 3. What this project actually does

Four things, each of which you can run:

1. **A pipeline** that pulls real solar wind measurements from NASA, streams them through a
   message bus into a database, transforms them into model-ready features with 18 automated
   quality tests, and refuses to proceed if the data is bad.

2. **A forecasting model** that predicts Dst 1 to 6 hours ahead. Its distinguishing feature
   is that the physics of the ring current is built into its structure, not left for it to
   learn. It also reports calibrated confidence intervals.

3. **Orchestration** in Dagster, so the pipeline runs on a schedule with full lineage,
   test results, and history visible in a browser.

4. **Two dashboards**: a live storm nowcast (the forecast with its uncertainty band and
   alert level), and a warehouse view (what is in the data, how clean it is, storm history).

---

## 4. Results on real NASA data

Trained and tested on **87,371 hours of real NASA OMNI data spanning 2014 to 2024**
(10 years, 1,999 storm hours, worst storm -234 nT). Chronological 70/15/15 split.

| Model | RMSE (nT) | Storm RMSE (nT) | What it is |
|---|:---:|:---:|---|
| **Physics-informed hybrid (this project)** | **8.16** | **19.89** | Neural network with the ring-current equation built in |
| O'Brien-McPherron | ~9.8 | ~22.5 | The classical physics equation alone, no learning |
| Persistence | ~9.4 | ~26.0 | "Nothing will change" |
| GRU (black box) | ~12.7 | ~26.7 | Standard neural network, no physics |

**Uncertainty calibration: 88.5% coverage against a 90% target.** When the model says it is
90% confident, it is right about 88.5% of the time. Close to honest.

Three things worth noticing:

- **The physics prior earns its place.** The hybrid beats the black-box GRU by a wide margin
  on the same data with the same inputs. Building in what we already know about the ring
  current works better than making the network rediscover it.
- **The black box loses to persistence.** A standard neural network performs *worse* than
  assuming nothing changes. This is a useful humility check on "just throw a neural net at it."
- **Storm RMSE is roughly 2.5x the overall RMSE.** The hard hours are much harder. This is
  consistent with the research literature, where extreme events remain the open problem.

*(Numbers vary slightly between runs, since training is not perfectly deterministic.)*

---

## 5. Quick start

**Requirements:** Python 3.10 or newer. That is it.

```bash
# 1. Install
pip install -e .

# 2. See the live storm dashboard immediately (uses a built-in simulator, no internet needed)
python -m heliostream serve --demo
# open http://127.0.0.1:8000
```

That works instantly because trained models can be regenerated in a couple of minutes:

```bash
python -m heliostream train --source synthetic --model hybrid --epochs 35
python -m heliostream evaluate --model hybrid
```

**To use real NASA data** (needs internet, no account or key required):

```bash
pip install hapiclient
python -m heliostream train --source omni --start 2014-01-01 --stop 2024-01-01 --model hybrid
python -m heliostream evaluate --model hybrid
```

The first fetch downloads about 10 years of hourly data and takes a few minutes.

> **About the simulator.** The project includes a physics-based synthetic data generator so
> that everything runs offline with zero setup. It is genuinely useful for development and
> testing, but it is **not** a substitute for real results. Every number in section 4 comes
> from real NASA data. Be suspicious of any project that only ever shows you simulated results.

---

## 6. The four things you can run

### A. The storm nowcast dashboard

```bash
python -m heliostream serve --demo     # simulated feed, works offline
python -m heliostream serve            # live NOAA feed, needs internet
```
Shows the incoming solar wind, the 1 to 6 hour Dst forecast with its 90% confidence band,
and a storm alert level. Open http://127.0.0.1:8000.

### B. The data pipeline

```bash
cd data_engineering
pip install -r requirements.txt && pip install -e ..

# Runs the whole thing: fetch -> bus -> database -> transform -> test -> quality gate
python -m heliostream_de run-batch --source omni --start 2014-01-01 --stop 2024-01-01

# Then train the model directly off the warehouse
python -m heliostream_de train --model hybrid --epochs 40
```
Use `--source synthetic --hours 26280` instead if you want it to run offline.

### C. Orchestration in Dagster

```bash
cd data_engineering/orchestration
pip install -e . && pip install -e ../..
dagster dev      # open http://localhost:3000, click Lineage, then Materialize all
```
Shows the full dependency graph, runs each step in order, displays every test result, and
attaches the model's metrics to the trained-model asset so you can track them over time.

### D. The warehouse dashboard

```bash
cd data_engineering
streamlit run dashboard/warehouse_app.py     # open http://localhost:8501
```
Storm history over the whole record, severity breakdown, storm frequency per month (the
solar cycle is clearly visible), feed coverage, and the data-quality gate.

**Windows note:** use `$env:VAR="value"` instead of `export VAR=value`, and semicolons rather
than colons in `PYTHONPATH`. The `make` shortcuts will not exist unless you use Git Bash or WSL.

---

## 7. How it works, layer by layer

```
  NASA OMNI (history)  ──┐
  NOAA SWPC (live)     ──┼──▶  producer  ──▶  message bus  ──▶  consumer  ──▶  raw.solar_wind
  built-in simulator   ──┘                    (Kafka/file)      (validate)      (DuckDB)
                                                                                     │
                                                                                     ▼
                                                                          dbt: staging -> features
                                                                          (+ 18 automated tests)
                                                                                     │
                                                                                     ▼
                                                                            quality gate
                                                                                     │
                                                                                     ▼
                              GRU reads last 24h  ──▶  predicts physics terms  ──▶  ODE rollout
                                                                                     │
                                                                                     ▼
                                                                     conformal calibration
                                                                                     │
                                                                                     ▼
                                                                    forecast + honest interval
```

### The interesting part: how physics gets inside the model

A normal neural network would take the solar wind and directly output a number. This one
does something different.

Physicists have known since 1975 roughly how the ring current behaves. It follows an
equation with two pieces:

```
   rate of change of the ring current  =  injection  -  decay
```

Meaning: energy pours in while the solar wind drives it (the injection term Q), and it
leaks away on its own with a characteristic timescale (the decay term tau). This is the
**O'Brien-McPherron** equation.

Instead of predicting Dst directly, Heliostream's network predicts **Q and tau**, the two
physical quantities. Then the code *integrates the equation forward* through time to produce
the forecast trajectory.

Why this is better:

- The forecast is **physically shaped by construction**. It cannot produce a nonsensical
  trajectory, because storms build and recover the way the equation says they do.
- The network only has to learn the hard part (how strongly this particular solar wind
  drives the ring current), not the easy part (that recovery is exponential).
- It **degrades gracefully**. When inputs are noisy or missing, the physics keeps the forecast
  sensible, whereas a black box can produce anything.

A small, deliberately bounded "residual" correction lets the network fix systematic errors in
the 1975 equation without being able to ignore the physics entirely. The whole thing, ODE
included, is written in PyTorch and is differentiable, so training adjusts the network
*through* the physics.

---

## 8. Two real bugs found in NASA's own data

This is the part worth reading if you care about data engineering.

The pipeline's automated tests were written against the simulator, where everything was
clean. The moment they met real NASA data, **two tests went red**. Both were informative,
and in opposite ways.

### Bug 1: physically impossible magnetic fields (a real defect)

The test `assert_bz_within_bt` encodes a physical fact: a component of a vector cannot be
longer than the vector itself, so |Bz| must be less than or equal to Bt.

**It found 11 hours in 87,371 where NASA's data violated this.**

The cause is not a NASA error exactly. OMNI computes the field magnitude and the individual
components from separately-averaged, separately-rounded quantities. During hours when the
field is very weak, the rounding can push the component slightly past the magnitude.

**Fix:** the feature model now clamps |Bz| to Bt, restoring physical consistency for the
features that depend on field direction. The data is not discarded, just made self-consistent.

### Bug 2: a 43-hour data gap (not a defect, a bad assumption)

The continuity check failed because it found a 43-hour gap where no data existed at all.

Investigating showed 27 gaps across 10 years, totalling 277 missing hours. That is **99.68%
coverage**. Four of the largest gaps clustered in a two-week window in November 2014,
suggesting a specific spacecraft issue.

But here is the thing: **this was not a data problem. It was a bad threshold.** The check
was set to fail on any gap longer than 24 hours, a limit chosen against simulated data that
had no gaps at all. Real spacecraft telemetry always has dropout. Failing on its existence is
wrong.

**Fix:** the gate is now gap-*aware* rather than gap-*intolerant*. It reports gap count,
missing hours, and largest outage as information, and fails only when coverage drops below
98% or a single outage exceeds 72 hours, which would signal a genuinely broken feed. This was
verified to still fail correctly on a simulated 10-day outage, because a check that always
passes is worthless.

### The lesson

Synthetic data validates your **code**. Real data validates your **assumptions**. One of these
was a genuine defect needing a fix; the other was my own threshold being naive. You cannot tell
which is which until you point the thing at reality.

---

## 9. Repository map

```
heliostream/
├── heliostream/                 THE MODEL
│   ├── models/physics.py          the O'Brien-McPherron ring-current equation
│   ├── models/hybrid.py           the physics-informed network (the main contribution)
│   ├── models/baseline.py         the black-box GRU, for comparison
│   ├── data/synthetic.py          the offline simulator
│   ├── data/omni.py               real NASA historical data (via CDAWeb HAPI)
│   ├── data/noaa_live.py          real NOAA live feed
│   ├── data/features.py           feature engineering and time-based windowing
│   ├── train.py                   training loop, storm weighting, model registry
│   ├── evaluate.py                metrics, conformal calibration, baselines, plots
│   ├── backtest.py                walk-forward validation across time
│   ├── serve.py                   the live forecast API
│   └── cli.py                     command line interface
├── dashboard/index.html         the storm nowcast dashboard
├── data_engineering/            THE PIPELINE
│   ├── heliostream_de/            producer, bus, consumer, warehouse, quality gate
│   ├── dbt/                       SQL transformations + 18 tests
│   ├── dashboard/                 the warehouse dashboard (Streamlit)
│   ├── orchestration/             Dagster assets, jobs, schedule
│   ├── docker-compose.yml         Redpanda + console + producer + consumer
│   └── tests/                     bus, warehouse, and feature-parity tests
├── heliostream_omni.ipynb       real-data notebook (runs in Google Colab)
└── tests/                       model tests (physics, shapes, calibration)
```

Run the tests with `python -m pytest tests/ -q` (model) and the same inside
`data_engineering/` (pipeline).

---

## 10. Honest limitations

Stated plainly, because a portfolio project that claims no weaknesses is not believable.

**The forecast horizon is softer than "6 hours" sounds.** The satellite at L1 only buys about
one hour of genuine lead time. Beyond that, the model assumes the incoming solar wind stays as
it is. So the 2 to 6 hour forecasts lean partly on persistence rather than on actually seeing
what is coming. True multi-hour warning would require watching the Sun itself with coronagraphs,
which is out of scope here.

**This is not a MagNet leaderboard result.** NASA and NOAA ran a public competition on this exact
task. Their benchmark scored 15.2 nT and the winners reached about 11 nT. Our 8.16 nT is *not*
comparable: we use hourly definitive OMNI data (cleaner than the real-time feed they scored on),
a different test split, and a different target formulation. The honest claim is "in the ballpark
of published skill," not "we won." A real leaderboard number would require running inside their
exact data and scoring harness.

**Extreme storms are a thin sample.** In 10 years there are very few severe (below -200 nT) hours.
Any claim about extreme-event performance rests on a small number of events. This is genuinely
the open problem in the field, not a shortcoming unique to this project.

**Definitive data is easier than live data.** OMNI is carefully reprocessed after the fact. The
real-time NOAA feed is noisier, gappier, and occasionally switches which spacecraft it uses.
Operational skill would be lower than the numbers above.

**Conformal calibration assumes exchangeability.** Forecasting a later time period technically
violates that assumption, which is likely why coverage lands at 88.5% rather than exactly 90%.

**DuckDB is single-writer.** Fine for a laptop, but you cannot have Dagster and the streaming
consumer writing at once. A production version would use Postgres.

**The Kafka path is less exercised than the file-based one.** It is wired up and runs under Docker
Compose, but the file-backed bus is what the automated tests use.

---

## 11. Credits and references

**Data sources**
- NASA/GSFC OMNI, via the CDAWeb HAPI interface: https://cdaweb.gsfc.nasa.gov/hapi/
- NOAA Space Weather Prediction Center real-time solar wind: https://services.swpc.noaa.gov/

**Science**
- O'Brien, T. P., and McPherron, R. L. (2000). *An empirical phenomenological model for the
  ring current.* Journal of Geophysical Research, 105(A4). This is the equation inside the model.
- Burton, R. K., McPherron, R. L., and Russell, C. T. (1975). The original ring-current model.
- Nair et al. (2023). *MagNet: A Machine Learning Competition for Real-Time Geomagnetic
  Forecasting.* Space Weather. The benchmark this work is positioned against.

**Built with** PyTorch, DuckDB, dbt, Dagster, Kafka/Redpanda, FastAPI, Streamlit, Altair.

**AI assistance disclosed.** This project was built with substantial help from Claude (Anthropic),
including the model architecture, pipeline, and this documentation. The results reported here were
produced by running the code on real NASA data, not estimated or simulated.

**Licence:** MIT. See [LICENSE](LICENSE).
