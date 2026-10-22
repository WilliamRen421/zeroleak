# zeroleak

> **Status: draft, not released.** The code runs and its tests pass, but the
> licence, the author list, the archive DOI and the public data registry are
> still to be settled, and the manuscript claims no repository URL. The full
> five-body run needs `ZEROLEAK_DATA_DIR` pointing at the external archives
> listed in `DATA.md`; only the Volve body ships with this directory.

A five-layer pipeline for predicting permeability from porosity alone, in which
each layer's information set is a **type in its signature** rather than a promise
in a paper.

The pipeline is the one used in the manuscript *Cutpoints transfer, intercepts do
not: a porosity-only decomposition of cross-field permeability transfer*. It
fits one power law per porosity phase, keeps the phase boundaries as a
configuration value, and refuses to let the model see permeability when it needs
permeability, or to let the fold layer choose a fold using the values it will be
tested on.

## The five layers

| layer | receives | cannot see | enforced by |
|---|---|---|---|
| `data` | a file on disk | — | the only module that opens a measurement file |
| `folds` | `Porosity`, `Depth`, a rule, a seed | `Permeability` | no permeability parameter exists |
| `model` | `Porosity`, `Permeability`, `FoldLabels`, `Cutpoints` | the fold boundaries | `FoldLabels` carries labels, not edges |
| `metrics` | `Predictions`, `Truth` | the model, any cutpoint | signature holds two arrays and nothing else |
| `report` | exported CSV | the model, numpy | imports neither |

A call that crosses a boundary raises `TypeError` before the body runs, and the
message names the information set that was about to cross:

```python
>>> assign_folds(body, body.groups)
TypeError: assign_folds(): 'phi' must be Porosity, got Body.
Porosity carries porosity values and nothing else;
Body carries porosity, permeability and depth read from one file.
```

`tests/test_layers.py` checks this at the syntax-tree level as well: `model.py`
contains no `quantile`, `percentile` or `median` call, `report.py` imports neither
numpy nor the model, and only `data.py` and `report.py` open a file. Those are
properties of the code, not of its documentation.

## Quick start

```bash
python -m zeroleak.run --config configs/base.yaml     # needs no download
python -m pytest -q                                   # 30 tests, no download
```

The shipped body is Volve 15/9-19A, 557 core plugs, together with its expected
output in `expected/`. The run reproduces the manuscript's Table 1 row exactly:

| strategy | held-out R² | RMSE/σ | samples |
|---|---|---|---|
| prior cutpoints 0.10/0.20 | 0.7207 | 0.5285 | 557 |
| target's own terciles | 0.7145 | 0.5343 | 557 |

With the full archives present, the other four rows are checked the same way:

```bash
export ZEROLEAK_DATA_DIR=/path/to/archives   # layout in DATA.md
python -m zeroleak.run --config configs/full.yaml
python -m pytest -q                          # 40 tests, five bodies
```

All five rows of Table 1 reproduce to the four decimals printed there, on every
numpy version tried (2.4.4 and 2.5.3):

| body | R² prior | R² terciles | samples |
|---|---|---|---|
| SPE 10 Tarbert | 0.7355 | 0.7336 | 100,000 |
| SPE 10 Upper Ness | 0.6634 | 0.6641 | 100,000 |
| Volve 15/9-19A | 0.7207 | 0.7145 | 557 |
| Brent 211/29 | 0.7430 | 0.7143 | 586 |
| Piper 15/17 | 0.4868 | 0.4642 | 4,428 |

## What one command writes

For each body and strategy: a CSV of per-sample predictions, a CSV of summary
metrics, and a JSON record of the configuration, the data file's SHA-256, the
numpy version and the git commit. Then `index.csv`, rebuilt from the exported
metric files rather than from memory, so an index that disagrees with its own
predictions is not possible without the files disagreeing too.

```
expected/
  volve_15_9_19a__prior.pred.csv
  volve_15_9_19a__prior.metrics.csv
  volve_15_9_19a__prior.config.json
  volve_15_9_19a__target_terciles.*
  index.csv
```

## Why the cutpoints are a separate type

`Cutpoints` has no `estimate`, `fit` or `terciles` method, so the model layer has
no path from training data to a boundary even if someone wanted one. Estimation
lives in `strategies.py`, where it takes `Porosity` and nothing else — legal,
because porosity is available at a new field before any core is cut. The FZI
criterion does the opposite: it partitions by a function of the permeability it
predicts, and no function in this package can express that, because no signature
admits both.

The distinction is the point of the design. A rule that may look at the target's
porosity is a strategy; a rule that must look at the target's permeability is a
leak, and the difference is visible in an argument list.

## Layout

```
zeroleak/
  types.py          the nine carriers and the runtime guard
  config.py         one YAML, validated before anything is loaded
  strategies.py     the only place a boundary may be estimated
  run.py            the single command
  layers/           data, folds, model, metrics, report
configs/base.yaml   shipped body
configs/full.yaml   all five, reads $ZEROLEAK_DATA_DIR
data/               the 557-plug Volve subset
expected/           output of the shipped configuration
scripts/            provenance registry: source, licence, SHA-256
tests/              40 tests: identity, headline, sign, layers, reproduction
```

## What the package does not do

No interactive interface, no database, no service. Each would add surface area
without changing what can be checked, and an interface would invite unrecorded
configurations. Every run writes its configuration beside its output so that a
number can be traced to a command and a commit.

It is not a tool for producing porosity–permeability transforms. It is an
arrangement in which the information set of every transform is visible in the
call signature, and therefore auditable by a reader who does not trust the prose.

## Data and licence

Code: MIT (see `LICENSE`). The shipped Volve subset is redistributed under the
Equinor Open Data Licence; see `DATA.md` for the source, licence and SHA-256 of
every input, and `scripts/fetch_public_data.py --verify` to check them.
