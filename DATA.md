# Data

Every input is a public archive. None of it was measured for this work, and none
of the licences is ours to relax.

## The five bodies

| body | file | rows used | source | licence | shipped |
|---|---|---|---|---|---|
| Volve 15/9-19A (subset) | `data/volve_15_9_19a.csv` | 557 | Equinor Volve data set, CPOR / CKHG / DEPTH | Equinor Open Data Licence | **yes** |
| Volve 15/9-19A (full) | `volve_core_full.csv` | 557 of 729 | Equinor Volve data set | Equinor Open Data Licence | no |
| SPE 10 Tarbert | `spe10/spe_phi.dat` + `spe_perm.dat` | 100,000 of 453,515 in window (462,000 cells) | SPE Comparative Solution Project 10, Christie & Blunt (2001) | distributed with the benchmark for research use | no |
| SPE 10 Upper Ness | same archive, z = 35–84 | 100,000 of 613,220 in window (660,000 cells) | as above | as above | no |
| Brent 211/29 | `brent/cgg_phik_211-29.csv` | 586, 5 wells | NSTA external DataReleases, CGG CoreAnalysis release | NSTA data release terms | no |
| Piper 15/17 | `piper/core_15-17.csv` | 4,428, 25 wells | NSTA external DataReleases, CGG CoreAnalysis release | NSTA data release terms | no |

The Brent and Piper extracts are two- and three-column projections of one
592,838-sample Access table. The raw table is not mirrored here.

## Checksums

`scripts/fetch_public_data.py --verify` recomputes these.

| file | SHA-256 |
|---|---|
| `data/volve_15_9_19a.csv` | `89a3d45fc836814bd9377bee353a73b5a44496bc5cfe31544b2a4282085785d9` |
| `volve_core_full.csv` | `c131154a2e7cafdc8200db8018480be77c4c7b29353a69677a328c439d10d915` |
| `spe10/spe_phi.dat` | `53f95e004a0803b04293539e0b1d5f82f2d25d341ca300cb117f1414d9dc7d8a` |
| `spe10/spe_perm.dat` | `7e5d613bf44dabaec2ed2dd7ebe87e79515bb77ac071dad9d33823efc37c7914` |
| `brent/cgg_phik_211-29.csv` | `435992f91fb2b97cb98b5ddccf267b2bd6d37202a8fabb3bb929270589cb867b` |
| `piper/core_15-17.csv` | `ec25a2d1a53d2d649e999f0c7db91bd2e8cc2fba8517e8ffc2fc335a1155d303` |

## Layout expected by `configs/full.yaml`

```
$ZEROLEAK_DATA_DIR/
  spe10/spe_phi.dat
  spe10/spe_perm.dat
  volve_core_full.csv
  brent/cgg_phik_211-29.csv
  piper/core_15-17.csv
```

## Two conventions that decide numbers

**Porosity is a fraction.** The archives store percent; `phi_scale: 100` divides
it. The window is 0.02 ≤ φ ≤ 0.45 with k > 0 everywhere.

**The permeability column is Kx.** `spe_perm.dat` is stored as `[Kx | Ky | Kz]`
blocks and only the first block is read. Reading it as interleaved triples
(`perm[0::3]`) produces a plausible-looking but wrong body.

## The shipped subset

`data/volve_15_9_19a.csv` is the Volve core table restricted to the window, with
three columns kept (depth, φ, k) and rows sorted by depth. It is a derived
extract, not the archive; the archive is 729 rows and is linked above.
