"""Command line: ``python -m sciglass_process <sciglass_dir> --out out/``."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .dataset import code_dictionary, coverage, process_by_glass, process_by_record, read_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sciglass_dir", help="clone of github.com/drcassar/SciGlass")
    ap.add_argument("--out", default="out", help="output directory")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    arhiv = read_table(args.sciglass_dir, "ARHIV", usecols=["KOD", "INFO", "NUM_GLASS"])
    rec = process_by_record(arhiv)
    scigk = read_table(args.sciglass_dir, "SciGK",
                       usecols=["KOD", "GLASNO", "TG", "DENSITY", "ND300", "MOD_UNG", "ANY_TEC", "TLiq"])
    glass = process_by_glass(rec, scigk)
    cov = coverage(rec, glass)

    rec.to_csv(out / "process_by_record.csv", index=False)
    glass.drop(columns=["melt_raw", "anneal_raw", "notes", "unparsed"]).to_csv(
        out / "process_by_glass.csv", index=False)
    code_dictionary(args.sciglass_dir).to_csv(out / "codes.csv", index=False)
    cov.to_csv(out / "coverage.csv", index=False)
    print(cov.to_string(index=False))
    print(f"\nwrote {out}/ in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
