"""Does melting history help predict glass properties beyond composition?

    python examples/process_benefit.py ../SciGlass            # Tg, density, refractive index
    python examples/process_benefit.py ../SciGlass TG         # one property

For each property: the same glasses, the same model, two feature sets.

* composition: full mol% composition (the 17 major oxides in ``SciGK`` plus the
  60 most common minor components in ``MolPc``, the rest pooled as "other");
  only glasses whose composition sums to 100 +/- 2 mol% are used
* composition + melting: the above plus the decoded melting temperature
  (min, max), total melting time, number of melting stages, vessel, casting and
  quenching codes

Annealing temperature is left out on purpose: it is usually chosen from the
glass's known Tg and would leak the target. Cross-validation is grouped by
source record (``KOD``), so each test fold holds only papers the model never saw.
A control run shuffles the melting columns across rows: if it scores like
"composition only", the gain is not an artefact of adding columns.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from sciglass_process.dataset import process_by_record, read_table

# mol% columns in SciGK; RO and FemOn are aggregates of other columns and are left out
OXIDES = ["SIO2", "AL2O3", "B2O3", "CAO", "K2O", "NA2O", "PBO", "Li2O", "MgO", "SRO",
          "BAO", "ZNO", "P2O5", "GEO2", "ZRO2", "TIO2", "TEO2"]
N_MINOR = 60
NUMERIC = ["melt_temp_max_C", "melt_temp_min_C", "melt_time_total_h", "n_melt_stages"]
CODES = ["vessel", "casting", "quenching"]
# property -> (plausible range, unit)
PROPERTIES = {"TG": ((100, 1500), "K"), "DENSITY": ((1.0, 9.0), "g/cm3"), "ND300": ((1.2, 3.5), "")}


def compositions(sciglass_dir: str, props: list[str]) -> tuple[pd.DataFrame, list[str]]:
    scigk = read_table(sciglass_dir, "SciGK", usecols=["KOD", "GLASNO", *props, *OXIDES])
    for c in ["KOD", "GLASNO", *props, *OXIDES]:
        scigk[c] = pd.to_numeric(scigk[c], errors="coerce")
    scigk[OXIDES] = scigk[OXIDES].fillna(0.0)

    minor = read_table(sciglass_dir, "MolPc")
    minor["GlasNo"] = pd.to_numeric(minor["GlasNo"], errors="coerce")
    minor["pValue"] = pd.to_numeric(minor["pValue"], errors="coerce")
    top = minor["pID"].value_counts().index[:N_MINOR]
    wide = minor[minor["pID"].isin(top)].pivot_table(index="GlasNo", columns="pID", values="pValue", aggfunc="sum")
    wide.columns = [f"m_{c}" for c in wide.columns]
    rest = minor[~minor["pID"].isin(top)].groupby("GlasNo")["pValue"].sum().rename("other")

    comp_cols = [*OXIDES, *wide.columns, "other"]
    df = scigk.merge(wide, left_on="GLASNO", right_index=True, how="left").merge(
        rest, left_on="GLASNO", right_index=True, how="left")
    df[comp_cols] = df[comp_cols].fillna(0.0)
    return df[df[comp_cols].sum(axis=1).between(98.0, 102.0)], comp_cols


def melting(sciglass_dir: str) -> pd.DataFrame:
    proc = process_by_record(read_table(sciglass_dir, "ARHIV", usecols=["KOD", "NUM_GLASS", "INFO"]))
    proc = proc[proc["qc_flag"].isna() & proc["melt_temp_max_C"].notna()].copy()
    for c in CODES:
        keep = proc[c].value_counts().index[:15]
        proc[c] = proc[c].where(proc[c].isin(keep), "rare").fillna("none").astype("category").cat.codes
    return proc[["KOD", *NUMERIC, *CODES]]


def cv_errors(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cat_mask=None) -> tuple[float, float]:
    errs = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=0,
                                          categorical_features=cat_mask)
        m.fit(X[tr], y[tr])
        errs[te] = m.predict(X[te]) - y[te]
    return float(np.sqrt(np.mean(errs ** 2))), float(np.mean(np.abs(errs)))


def run(df: pd.DataFrame, comp: list[str], prop: str) -> list[tuple]:
    (lo, hi), unit = PROPERTIES[prop]
    d = df[df[prop].between(lo, hi)].reset_index(drop=True)
    y, groups = d[prop].to_numpy(float), d["KOD"].to_numpy()
    full = comp + NUMERIC + CODES
    cat = np.array([c in CODES for c in full])
    shuffled = d[full].copy()
    perm = np.random.default_rng(0).permutation(len(d))
    shuffled[NUMERIC + CODES] = shuffled[NUMERIC + CODES].to_numpy()[perm]
    base = cv_errors(d[comp].to_numpy(float), y, groups)
    proc = cv_errors(d[full].to_numpy(float), y, groups, cat)
    ctrl = cv_errors(shuffled.to_numpy(float), y, groups, cat)
    return [(prop, unit, len(d), d["KOD"].nunique(), base, proc, ctrl)]


def main() -> None:
    sciglass_dir = sys.argv[1] if len(sys.argv) > 1 else "../SciGlass"
    props = sys.argv[2:] or list(PROPERTIES)
    comp_df, comp = compositions(sciglass_dir, props)
    df = comp_df.merge(melting(sciglass_dir), on="KOD", how="inner")
    print("5-fold CV grouped by source record; RMSE (MAE)\n")
    print("| property | glasses | papers | composition only | + melting history | change | control (shuffled) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for prop in props:
        for p, unit, n, k, base, proc, ctrl in run(df, comp, prop):
            fmt = (lambda v: f"{v:.1f}") if unit == "K" else (lambda v: f"{v:.4f}")
            change = 100.0 * (proc[0] - base[0]) / base[0]
            label = f"{p} ({unit})" if unit else p
            print(f"| {label} | {n:,} | {k:,} | {fmt(base[0])} ({fmt(base[1])}) | "
                  f"{fmt(proc[0])} ({fmt(proc[1])}) | {change:+.1f}% | {fmt(ctrl[0])} ({fmt(ctrl[1])}) |")


if __name__ == "__main__":
    main()
