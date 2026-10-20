# sciglass-process

[![CI](https://github.com/saeidi-mohammadreza/sciglass-process/actions/workflows/ci.yml/badge.svg)](https://github.com/saeidi-mohammadreza/sciglass-process/actions/workflows/ci.yml)

**Recover the melting and annealing history that every glass-ML pipeline throws away.**

SciGlass is the largest open glass database (422,949 glasses from 39,949 literature sources). Machine-learning work on it reads compositions and properties from the flattened `SciGK` table. But each source record in the `ARHIV` table also starts with a compact coded header that says *how the glasses were made*: crucible, melting temperature and time, atmosphere, casting, annealing temperature and time, cooling rate. Nobody parses it. This package does.

```
#m 1 pc 2. 1200-1250,a 1 500 3. 1,x 130 1 me 1. old,r 'Author A.' ...#
  │  │    └─ melted at 1200–1250 °C      └─ annealed at 500 °C for 1 h
  │  └─ in a platinum crucible
  └─ melting clause
```

| vessel | melt_temp_min_C | melt_temp_max_C | melt_time_total_h | anneal_temp_max_C | anneal_time_h |
| --- | ---: | ---: | ---: | ---: | ---: |
| pc | 1200 | 1250 | | 500 | 1 |

## What is in there

Counts on the open release ([drcassar/SciGlass](https://github.com/drcassar/SciGlass), DOI [10.5281/zenodo.8287159](https://doi.org/10.5281/zenodo.8287159)), produced by `python -m sciglass_process` and pinned by `tests/test_real_data.py`:

| | source records | glasses |
| --- | ---: | ---: |
| total | 39,949 | 422,949 |
| with a melting clause | **27,519 (69%)** | 271,150 |
| with a melting temperature | 19,915 | 197,733 |
| with a melting time | 15,341 | |
| with a crucible / vessel | 17,924 | |
| with an annealing temperature | 5,546 | |
| **with Tg *and* a melting temperature** | | **64,732** |
| with Tg, melting temperature and annealing temperature | | 10,765 |

So a composition → Tg model can be extended to composition **+ process** → Tg on about 65 thousand glasses, without collecting a single new measurement.

Most common melting vessels (codes and wording are SciGlass's own legend, table `DT`):

| code | meaning | records |
| --- | --- | ---: |
| `pc` | Melting in a platinum crucible | 9,693 |
| `ac` | Melting in alumina crucibles | 2,827 |
| `sc` | Melting in silica crucibles | 1,381 |
| `ptrd` | Melting in a platinum-rhodium crucible | 1,289 |
| `porc` | Melting in porcelain crucibles | 895 |
| `gcc` | Melting in glassy carbon crucibles | 347 |

Median melting temperature 1,300 °C (IQR 1,000–1,450 °C); median annealing temperature 420 °C.

## Install and run

```bash
pip install -e .
bash scripts/fetch_sciglass.sh ../SciGlass          # ~190 MB, ODbL
python -m sciglass_process ../SciGlass --out out  # ~12 s
```

Outputs in `out/`:

| file | rows | content |
| --- | --- | --- |
| `process_by_record.csv` | one per source record (`KOD`) | decoded columns + the raw clauses + a `qc_flag` |
| `process_by_glass.csv` | one per glass (`GLASNO`) | process columns joined to Tg, density, n<sub>d</sub>, Young's modulus, CTE, liquidus |
| `codes.csv` | 81 | SciGlass's legend for every `m`/`a` code |
| `coverage.csv` | | the table above |

As a library:

```python
from sciglass_process import parse_process

p = parse_process("#m 1 clp 2 1500 3. 3 8 cw 0 'and remelting' 2 1500 3 2 2 1600 3. 1#")
p.vessel, p.melt_temp_max_C, p.melt_time_total_h, p.casting   # ('clp', 1600.0, 6.0, 'cw')
[(s.temp_max_C, s.time_h) for s in p.stages]                 # [(1500, 3), (1500, 2), (1600, 1)]
```

## How the header is decoded

* The header is the text between the first two `#` of `ARHIV.INFO`; clauses are split on commas outside quotes.
* `m` (melting) and `a` (annealing) clauses are sequences of *field number → value* pairs. Field meanings come from SciGlass's `DT` table: `m 1` vessel, `m 1s` non-melt synthesis (sol-gel, vapour deposition), `m 1q` quenching, `m 2` temperature °C, `m 3`/`3m` time h/min, `m 4` atmosphere/stirring/remelting, `m 8` casting, `m 9` bubbling; `a 1` temperature °C, `a 1g`/`2g` K above/below Tg, `a 3` time, `a 4`/`4r` cooling.
* Repeated temperature/time pairs become separate melting stages (remelting schedules are common).
* Values handled: ranges `1200-1250`, tolerances `1100±20`, SciGlass's "approximately" mark, `0,5`, a trailing sentence dot, a marker glued to its value (`3.6`), quoted notes with or without the `0` marker.
* Nothing is dropped silently: anything not understood lands in `unparsed` (0.2% of melting clauses, mostly free text such as "temperatures up to 1000").
* `qc_flag` marks 58 records whose temperatures fall outside 150–2,300 °C (melting) or 50–1,500 °C (annealing), e.g. a `1000-13000` typo.

## Caveats

* The header describes the **source record**, so every glass in a record inherits the same process. Within-paper variation in melting conditions is not resolved.
* 93% of SciGlass compositions are *as batched*, not analysed; process conditions such as crucible material and melting time are exactly what drives the difference (volatilisation, crucible dissolution). Use the vessel and time columns to model or filter that noise.
* SciGlass data are released under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/). Derived tables you publish must keep that licence and attribution. This repository contains code only (MIT); no SciGlass data are redistributed.

## Tests

```bash
pytest tests/test_parser.py                                   # grammar, no data needed
SCIGLASS_DIR=../SciGlass pytest tests/test_real_data.py       # pins the counts above
```

## Background

This came out of a 306-source survey of public data for a batch-composition + process → property model for industrial glass (August 2026). The finding that made the project feasible was that the process half of the feature space, assumed missing from every public database, is already sitting in this header for 69% of SciGlass's sources.

Author: Mohammadreza Saeidi · M.Sc. Materials Engineering (IUST) · [LinkedIn](https://www.linkedin.com/in/mohammadreza-saeidi)
