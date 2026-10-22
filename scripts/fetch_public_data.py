"""Where every byte came from.

A registry, not a downloader.  Each entry records the source, the licence and the
SHA-256 of the file the pipeline actually reads, so that a reader who has the
archive can prove it is the same one this work used, and a reader who does not
can see exactly what is missing.

    python scripts/fetch_public_data.py --list
    python scripts/fetch_public_data.py --verify [--data-dir DIR]

`--verify` hashes whatever is present and reports the entries it cannot find.
Nothing is downloaded: three of these archives are tens of megabytes and two are
redistributed under terms that do not permit us to mirror them.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGISTRY: List[Dict[str, Optional[str]]] = [
    {
        "body": "Volve 15/9-19A (shipped subset)",
        "path": "data/volve_15_9_19a.csv",
        "rows": "557",
        "source": "Equinor Volve data set; CPOR, CKHG and DEPTH columns, windowed to 0.02-0.45 and k>0",
        "licence": "Equinor Open Data Licence (Volve release)",
        "url": "https://www.equinor.com/energy/volve-data-sharing",
        "sha256": "89a3d45fc836814bd9377bee353a73b5a44496bc5cfe31544b2a4282085785d9",
        "shipped": "yes",
    },
    {
        "body": "Volve 15/9-19A (full core table)",
        "path": "${ZEROLEAK_DATA_DIR}/volve_core_full.csv",
        "rows": "729 raw, 557 in window",
        "source": "Equinor Volve data set, core analysis export",
        "licence": "Equinor Open Data Licence (Volve release)",
        "url": "https://www.equinor.com/energy/volve-data-sharing",
        "sha256": "c131154a2e7cafdc8200db8018480be77c4c7b29353a69677a328c439d10d915",
        "shipped": "no",
    },
    {
        "body": "SPE 10 Model 2, porosity",
        "path": "${ZEROLEAK_DATA_DIR}/spe10/spe_phi.dat",
        "rows": "85 x 220 x 60 = 1,122,000 cells",
        "source": "SPE Comparative Solution Project 10 (Christie & Blunt, 2001), spe_phi.dat",
        "licence": "Distributed with the SPE 10 benchmark for research use; cite Christie & Blunt (2001)",
        "url": None,
        "sha256": "53f95e004a0803b04293539e0b1d5f82f2d25d341ca300cb117f1414d9dc7d8a",
        "shipped": "no",
    },
    {
        "body": "SPE 10 Model 2, permeability",
        "path": "${ZEROLEAK_DATA_DIR}/spe10/spe_perm.dat",
        "rows": "same grid, stored as [Kx | Ky | Kz]; only the Kx block is read",
        "source": "SPE Comparative Solution Project 10 (Christie & Blunt, 2001), spe_perm.dat",
        "licence": "Distributed with the SPE 10 benchmark for research use; cite Christie & Blunt (2001)",
        "url": None,
        "sha256": "7e5d613bf44dabaec2ed2dd7ebe87e79515bb77ac071dad9d33823efc37c7914",
        "shipped": "no",
    },
    {
        "body": "Brent 211/29",
        "path": "${ZEROLEAK_DATA_DIR}/brent/cgg_phik_211-29.csv",
        "rows": "586 in window, 5 wells",
        "source": "NSTA external DataReleases, CGG CoreAnalysis release; two-column extract (phi_pct, kh_md, well)",
        "licence": "NSTA data release terms; the raw Access table is not redistributed here",
        "url": "https://datanstauthority.blob.core.windows.net/external/DataReleases/",
        "sha256": "435992f91fb2b97cb98b5ddccf267b2bd6d37202a8fabb3bb929270589cb867b",
        "shipped": "no",
    },
    {
        "body": "Piper 15/17",
        "path": "${ZEROLEAK_DATA_DIR}/piper/core_15-17.csv",
        "rows": "4,428 in window, 25 wells",
        "source": "NSTA external DataReleases, CGG CoreAnalysis release; extract (Porosity_1, Hor_Permeability_1, WellName)",
        "licence": "NSTA data release terms; the raw Access table is not redistributed here",
        "url": "https://datanstauthority.blob.core.windows.net/external/DataReleases/",
        "sha256": "ec25a2d1a53d2d649e999f0c7db91bd2e8cc2fba8517e8ffc2fc335a1155d303",
        "shipped": "no",
    },
]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path: str, data_dir: str) -> str:
    if "$ZEROLEAK_DATA_DIR" in path and data_dir:
        path = path.replace("${ZEROLEAK_DATA_DIR}", data_dir)
    path = os.path.expandvars(path)
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Document and verify the provenance of every input file.")
    ap.add_argument("--list", action="store_true", help="print the registry")
    ap.add_argument("--verify", action="store_true", help="hash the files that are present")
    ap.add_argument("--data-dir", default=os.environ.get("ZEROLEAK_DATA_DIR", ""))
    args = ap.parse_args(argv)

    if not (args.list or args.verify):
        ap.print_help()
        return 0

    missing = 0
    if args.list:
        print(f"{'body':32s}{'rows':34s}{'shipped':8s}sha256")
        for e in REGISTRY:
            print(f"{e['body']:32s}{e['rows']:34s}{e['shipped']:8s}{e['sha256'][:16]}...")

    if args.verify:
        for e in REGISTRY:
            path = _resolve(e["path"], args.data_dir)
            if not os.path.exists(path):
                missing += 1
                print(f"MISSING  {e['body']:32s} {e['path']}")
                continue
            got = _sha256(path)
            ok = got == e["sha256"]
            print(f"{'OK      ' if ok else 'MISMATCH'} {e['body']:32s} {os.path.basename(path)}")
            if not ok:
                print(f"         expected {e['sha256']}\n         got      {got}")
        if missing:
            print(f"\n{missing} of {len(REGISTRY)} files are not on this machine; see DATA.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
