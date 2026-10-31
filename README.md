# zeroleak

> **The full run needs the external archives.** Only the Volve 15/9-19A body
> ships in this directory, and it ships in full; the other four need
> `ZEROLEAK_DATA_DIR` pointing at the archives listed in `DATA.md`. The Volve
> body reproduces `expected/volve_15_9_19a__prior.*` end to end. The six-page
> manual is in `docs/`.

A five-layer pipeline for predicting permeability from porosity alone, in which
each layer's information set is a **type in its signature** rather than a promise
in a paper.

The pipeline is the one used in the manuscript *Avoiding target-information
leakage in cross-field permeability prediction: prior phase boundaries,
admissible parameter ranges, and decomposition of prediction transfer loss*.
It fits one power law per porosity phase, keeps the phase boundaries as a
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

## The registry: what a declaration claims, what a signature allows

`requires` answers what a call may *receive*. It cannot answer what the method
*says* it uses, because that claim is prose. Registering each predictor with the
information set its description claims, and reading the implemented set off
`__requires__`, makes the two comparable in one table:

| predictor | declared | derived | status | available at target |
|---|---|---|---|---|
| `prior` | (nothing) | (nothing) | ok | yes |
| `target_terciles` | `porosity:target` | `porosity` | ok | yes |
| `free_exponent` | `permeability:train`, `porosity:train`, `folds:train`, `cutpoints:prior` | `permeability`, `porosity`, `folds`, `cutpoints` | ok | **no** |

Three `ok` rows are the intended result: the prior reads nothing, the tercile
rule reads porosity, and the fit reads permeability inside a fold. The check
earns its place by what it refuses, and two shapes fail:

- a predictor declared `porosity:target` whose signature also takes
  `Permeability` — the shape of the FZI family, described as a porosity
  criterion and computed from the permeability it predicts;
- any function that takes `Body`, because one legal type name carries porosity,
  permeability and depth at once. `CARRIES` already says so; the rule that a
  `Body` never leaves the data layer is a convention, and a convention is not a
  check.

Neither is caught by the types, because both signatures are legal. The
difference is written in the declaration, and comparing it with the signature is
all this module does — no parsing, no inference from a body. What the comparison
cannot check is provenance: a signature can say *permeability*, it cannot say
*whether that permeability is the target's*. `permeability:train` and
`permeability:target` are the same signature and different methods, which is why
the token is written out rather than inferred.

## The audit: the failure a signature cannot catch

Types certify what a function may see. They cannot certify that the held-out fold
is worth scoring, because a fold rule is a function of porosity and depth that
returns labels: a random split satisfies every check in `tests/` while leaving a
core plug 25 cm from the sample it is asked to predict. The information set is
clean and the evaluation is not.

`zeroleak audit` prints the declaration table first, then measures the split
instead of declaring it. For every sample it finds the distance to the nearest
sample in another fold and reports it in units of the median spacing between
adjacent samples:

| rule | nearest training sample | test samples within 1 spacing | R² prior | R² terciles | ΔR² | verdict |
|---|---|---|---|---|---|---|
| `depth_block` | 44.0 spacings | 0.7% | 0.7207 | 0.7145 | +0.0062 | blocked |
| `index_block` | 42.0 spacings | 1.1% | 0.7069 | 0.7029 | +0.0040 | blocked |
| `random_block` | 1.0 spacings | 86.0% | 0.724–0.734 | 0.7317 | +0.0002…+0.0054 | **interpolation** |

The geometry columns come from the default assignment, so they read the same
with or without `--seeds`; only the two right-hand columns open into ranges.

The last row is the one to read. It is admitted by every layer — the same
`FoldLabels`, the same signature — and it shrinks the cutpoint difference the
manuscript reports from $+0.0062$ to at most $+0.0054$ over five seeds while
adding up to $0.013$ to held-out R². Interpolation flatters this paper's own
claim, which is why the numbers it reports come from the certified rule.

```bash
python -m zeroleak audit --config configs/base.yaml            # ~1 s, no download
python -m zeroleak audit --config configs/base.yaml --seeds 5  # the range above
python -m zeroleak audit --config configs/base.yaml --strict   # exit 1 on either finding
```

## Quick start

Unzip, then run everything below from inside the `zeroleak-1.1.0/` directory that the
archive contains -- `pip install -e .` looks for `pyproject.toml` in the working directory,
and a shell sitting one level up reports that there is nothing to install.

```bash
pip install -e ".[test]"                              # Python 3.11, numpy 2.4.4
python -m zeroleak run --config configs/base.yaml     # needs no download
python -m zeroleak verify --config configs/base.yaml  # diff every cell against expected/
python -m pytest                                      # 66 passed, 14 skipped without the archives
```

The shipped body is Volve 15/9-19A, 557 core plugs, together with its expected
output in `expected/`. The run reproduces the manuscript's Table 4 row exactly:

| strategy | held-out R² | RMSE/σ | samples |
|---|---|---|---|
| prior cutpoints 0.10/0.20 | 0.7207 | 0.5285 | 557 |
| target's own terciles | 0.7145 | 0.5343 | 557 |

With the full archives present, the other four rows are checked the same way:

```bash
export ZEROLEAK_DATA_DIR=/path/to/archives   # layout in DATA.md
python -m zeroleak run --config configs/full.yaml
python -m pytest                             # 80 passed, five bodies
```

All five rows of Table 4 reproduce to the four decimals printed there:

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
out/base/
  volve_15_9_19a__prior.pred.csv
  volve_15_9_19a__prior.metrics.csv
  volve_15_9_19a__prior.config.json
  volve_15_9_19a__target_terciles.*
  index.csv
```

Nothing writes to `expected/`. That directory is the reference the manuscript's
numbers were read from, produced by a tagged commit; `zeroleak verify` runs the
configuration into `out/base/_verify` and then reports, file by file and cell by
cell, whether the result still equals it. A pipeline that could write to its own
reference would be checking itself against itself.

## The cutpoint scan and the leave-one-body-out check

The reported cutpoints are `(0.10, 0.20)`. A pair of numbers is not evidence, so
the package can refit the model at every pair on a grid and let a reader see
whether that pair is a point or a plateau:

```bash
python -m zeroleak scan     --config configs/full.yaml   # 951 cells, ~3.5 min
python -m zeroleak scan-loo --config configs/full.yaml   # < 1 s, reads the grid
```

`scan` fixes the grid before the data are read: `phi1` in [0.05, 0.20], `phi2` in
[0.10, 0.30], step 0.005, with `phi2 > phi1 + 0.02`, which leaves 951 evaluated
cells. Each cell is a complete five-fold run, not a single fit: the model is
refitted inside each training fold and scored on the held-out one, and the loss
is `R2_own - R2_abs`, so a positive value means the absolute cutpoints are worse
at that cell than the target's own terciles. It is single-threaded, so wall time
follows clock speed rather than core count; about five minutes on the machine the
paper was written on.

`scan-loo` reads the grid `scan` wrote and recomputes the optimum with each body
removed in turn, which is how Table S9 of the supplement is produced. The order
matters and the command says so rather than guessing: run it without a grid and
it exits with a message naming the command to run first.

What `scan-loo` cannot show is worth stating in the code as well as in the paper.
The criterion is "worst-body loss <= eps", so removing a body can only relax it
and "is this cell still inside the window" is true by construction. The check is
a statement about where the plateau sits and how wide it is — not about coverage.

```
<output_dir>/scan/
  cut_scan_report.txt   the grid, the argmin, the windows, the slices, the folds
  csv/grid_perbody.csv  951 rows x 5 bodies: the loss at every cell
  csv/grid_worst.csv    the worst body at every cell
  csv/slice_A.csv       phi2 fixed, phi1 swept
  csv/slice_B.csv       phi1 fixed, phi2 swept
  csv/robust_folds.csv  fold-offset sweep
  csv/boot.csv          1000 paired bootstrap resamples at (0.10, 0.20)
  csv/loo_argmin.csv    written by scan-loo
  scan.json             the grid, the offsets, the seed, the cell count
```

## The multi-seed check

The loss at the reported pair is not a single number: which subsample of a gridded
body is scored, and where the fold boundaries fall, are arbitrary and neither is a
property of the rock. The supplement quotes one standard deviation of that movement,
so the package recomputes it — at one pair, not on a grid, because a grid of standard
deviations answers a question nobody asked:

```bash
python -m zeroleak noise --config configs/full.yaml            # 20 draws, ~25 s
python -m zeroleak noise --config configs/base.yaml --draws 6  # 1 s, shipped body
```

Draw *i* takes subsample seed `seed + 1000 (i+1)` and its own fold offset, both from a
table in `zeroleak/noise.py`, so every draw redraws both choices at once. The offsets
stay inside **one period of the fold partition** — the grid folds into five slabs of
12 columns, so a shift of 60 cells returns the same partition; the cored well folds
into five depth blocks, so the period is a fifth of the cored interval — and none of
them is a multiple of the slab width, because such a shift only relabels the bins.

The twenty offsets are listed rather than generated. A sentence like "the boundaries
move by 1--59 cells" admits many sequences of twenty points; only one of them produced
the printed table, so the values are carried in the source. The published table
reproduces column for column, including the baseline column, which is draw one:

| body | baseline | mean | sd | min | max |
|---|---|---|---|---|---|
| SPE 10 Tarbert | -0.00206 | -0.00183 | 0.00026 | -0.00224 | -0.00124 |
| SPE 10 Upper Ness | +0.00032 | +0.00056 | 0.00015 | +0.00024 | +0.00079 |
| Volve 15/9-19A | -0.00630 | -0.00632 | 0.00272 | -0.01328 | -0.00201 |
| Brent 211/29 | -0.02863 | -0.02863 | 0.00000 | -0.02863 | -0.02863 |
| Piper 15/17 | -0.02262 | -0.02262 | 0.00000 | -0.02262 | -0.02262 |

The two grid bodies move with the subsample and barely with the offset; the cored well
is the other way round, having no subsample to draw. `--draws` accepts other counts,
but any count other than twenty switches to evenly spaced offsets, and the table above
then no longer applies.

The two well-folded bodies are reported as `fixed`, not as a standard deviation of
zero. Zero would claim that the number was checked not to move; the design cannot
support that, so the check says which three bodies it speaks for and stays quiet
about the other two.

`noise` also prints an anchor: the single-seed evaluation at the same pair, which is
what `scan` reports and what Table 4's tercile-minus-prior difference already is. If
the redraws did not bracket it, the two commands would not be scoring the same thing.

```
<output_dir>/noise/
  noise_report.txt    the twenty draws, the per-body spread, the anchor
  noise_results.json  the same numbers, plus every draw's loss
  noise.csv           draw, seed, grid cells, depth fraction, body, loss
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
  run.py            the default command: Table 4
  audit.py          fold-distance audit of the shipped bodies
  scan.py           951-cell cutpoint grid (Table 6, Fig. 5)
  scan_loo.py       leave-one-body-out recomputation (Table S9)
  noise.py          spread of the loss over redrawn subsamples and folds (Table S7)
  layers/           data, folds, model, metrics, report
configs/base.yaml   shipped body
configs/full.yaml   all five, reads $ZEROLEAK_DATA_DIR
data/               the 557-plug Volve subset
expected/           output of the shipped configuration
scripts/            provenance registry: source, licence, SHA-256
tests/              80 tests: identity, headline, sign, layers, grid, noise, reproduction
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

Code: MIT (see `LICENSE`). The shipped Volve body is redistributed under the
Equinor Open Data Licence; `DATA.md` carries the source, licence and SHA-256 of
every input, and `scripts/fetch_public_data.py --verify` checks a copy you have.

| body | source | in this repository |
|---|---|---|
| Volve 15/9-19A | Equinor open release, CPOR / CKHG / DEPTH | **yes, in full** |
| SPE 10 Tarbert, Upper Ness | public benchmark, Christie & Blunt (2001) | no; free to obtain |
| Brent 211/29, Piper 15/17 | NSTA DataReleases, CGG CoreAnalysis | no; NSTA Open User Licence |

**Volve ships in full.** `data/volve_15_9_19a.csv` holds every core plug inside
the analysis window — 557 of the archive's 729 rows, the remainder falling
outside 0.02 ≤ φ ≤ 0.45 or carrying no permeability — rather than a subsample of
them. That is why `run`, `verify` and 66 of the 80 tests need no download.

**SPE 10 is public.** Both units are distributed with the benchmark for research
use. `scripts/fetch_public_data.py` is a registry rather than a downloader: it
records the source, licence and SHA-256 of each file, and `--verify` re-hashes
whatever is present. Nothing is fetched, because three of these archives are
tens of megabytes and this repository does not mirror them.

**Brent and Piper are available under the NSTA Open User Licence.** Both are
two- and three-column projections of one CGG CoreAnalysis release published in
the NSTA's external DataReleases; the raw Access table is not redistributed
here. `DATA.md` gives the columns read, the rows, the SHA-256 of each extract
and where to download it.
