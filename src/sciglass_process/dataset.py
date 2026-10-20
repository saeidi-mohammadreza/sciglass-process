"""Build tidy process tables from a local copy of the open SciGlass release.

Expected layout (the plain-text conversion by D. R. Cassar,
https://github.com/drcassar/SciGlass, DOI 10.5281/zenodo.8287159)::

    <sciglass_dir>/sciglass/select/ARHIV.csv.zip
    <sciglass_dir>/sciglass/select/SciGK.csv.zip
    <sciglass_dir>/sciglass/property/DT.csv
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from .parser import parse_process

__all__ = ["read_table", "code_dictionary", "process_by_record", "process_by_glass", "coverage"]

# plausibility window for a melting temperature in degrees Celsius
MELT_T_RANGE = (150.0, 2300.0)
ANNEAL_T_RANGE = (50.0, 1500.0)


def _select(sciglass_dir: Path, name: str) -> Path:
    root = Path(sciglass_dir)
    for cand in (root / "sciglass" / "select" / name, root / "select" / name, root / name):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{name} not found under {root}")


def read_table(sciglass_dir: str | Path, table: str, usecols=None) -> pd.DataFrame:
    """Read ``ARHIV`` or ``SciGK`` (tab-separated, zipped) as strings."""
    path = _select(Path(sciglass_dir), f"{table}.csv.zip")
    with zipfile.ZipFile(path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(member) as fh:
            return pd.read_csv(fh, sep="\t", dtype=str, keep_default_na=False,
                               usecols=usecols, encoding="utf-8", encoding_errors="replace", low_memory=False)


def code_dictionary(sciglass_dir: str | Path) -> pd.DataFrame:
    """SciGlass's own legend for clause codes (table ``DT``): kind, field, code, text."""
    root = Path(sciglass_dir)
    for cand in (root / "sciglass" / "property" / "DT.csv", root / "property" / "DT.csv"):
        if cand.exists():
            dt = pd.read_csv(cand, dtype=str, keep_default_na=False, encoding="utf-8", encoding_errors="replace")
            dt = dt.rename(columns={"KOD": "kind", "NN": "field", "SNAME": "code", "NAME": "text"})
            dt = dt[dt["kind"].isin(["m", "a"])]
            for c in ("kind", "field", "code", "text"):
                dt[c] = dt[c].str.strip()
            return dt[["kind", "field", "code", "text"]].reset_index(drop=True)
    raise FileNotFoundError("property/DT.csv not found")


def process_by_record(arhiv: pd.DataFrame) -> pd.DataFrame:
    """One row per source record (``KOD``) with the decoded process columns."""
    rows = [parse_process(s).as_row() for s in arhiv["INFO"]]
    out = pd.DataFrame(rows)
    out.insert(0, "KOD", pd.to_numeric(arhiv["KOD"], errors="coerce").astype("Int64").to_numpy())
    out.insert(1, "n_glasses", pd.to_numeric(arhiv["NUM_GLASS"], errors="coerce").astype("Int64").to_numpy())
    lo, hi = MELT_T_RANGE
    bad_melt = out["melt_temp_max_C"].notna() & ~out["melt_temp_max_C"].between(lo, hi) | (
        out["melt_temp_min_C"].notna() & ~out["melt_temp_min_C"].between(lo, hi))
    alo, ahi = ANNEAL_T_RANGE
    bad_ann = out["anneal_temp_max_C"].notna() & ~out["anneal_temp_max_C"].between(alo, ahi)
    flags = pd.Series("", index=out.index)
    flags[bad_melt] += "melt_temp_implausible;"
    flags[bad_ann] += "anneal_temp_implausible;"
    out["qc_flag"] = flags.str.rstrip(";").replace("", None)
    return out


def process_by_glass(process: pd.DataFrame, scigk: pd.DataFrame,
                     properties: tuple[str, ...] = ("TG", "DENSITY", "ND300", "MOD_UNG", "ANY_TEC", "TLiq")) -> pd.DataFrame:
    """Attach the record-level process to every glass (``GLASNO``) via ``KOD``."""
    cols = ["KOD", "GLASNO", *[p for p in properties if p in scigk.columns]]
    g = scigk[cols].copy()
    g["KOD"] = pd.to_numeric(g["KOD"], errors="coerce").astype("Int64")
    g["GLASNO"] = pd.to_numeric(g["GLASNO"], errors="coerce").astype("Int64")
    for p in cols[2:]:
        g[p] = pd.to_numeric(g[p], errors="coerce")
    return g.merge(process, on="KOD", how="left", validate="many_to_one")


def coverage(by_record: pd.DataFrame, by_glass: pd.DataFrame | None = None) -> pd.DataFrame:
    """Headline counts: how much process information the header actually holds."""
    clean = by_record["qc_flag"].isna()
    rows = [
        ("source records", len(by_record), None),
        ("records with a melting clause", int(by_record["has_melt_clause"].sum()), None),
        ("records with an annealing clause", int(by_record["has_anneal_clause"].sum()), None),
        ("records with a melting temperature", int(by_record["melt_temp_max_C"].notna().sum()), None),
        ("records with a melting time", int(by_record["melt_time_total_h"].notna().sum()), None),
        ("records with a crucible / vessel", int(by_record["vessel"].notna().sum()), None),
        ("records with an annealing temperature", int(by_record["anneal_temp_max_C"].notna().sum()), None),
        ("records flagged implausible", int((~clean).sum()), None),
    ]
    df = pd.DataFrame(rows, columns=["item", "records", "glasses"])
    if by_glass is not None:
        g = by_glass
        df.loc[len(df)] = ("glasses (all)", None, len(g))
        df.loc[len(df)] = ("glasses with a melting clause", None, int(g["has_melt_clause"].fillna(False).astype(bool).sum()))
        df.loc[len(df)] = ("glasses with a melting temperature", None, int(g["melt_temp_max_C"].notna().sum()))
        if "TG" in g:
            df.loc[len(df)] = ("glasses with Tg", None, int(g["TG"].notna().sum()))
            df.loc[len(df)] = ("glasses with Tg and a melting temperature", None,
                               int((g["TG"].notna() & g["melt_temp_max_C"].notna()).sum()))
            df.loc[len(df)] = ("glasses with Tg, melting T and annealing T", None,
                               int((g["TG"].notna() & g["melt_temp_max_C"].notna() & g["anneal_temp_max_C"].notna()).sum()))
    return df
