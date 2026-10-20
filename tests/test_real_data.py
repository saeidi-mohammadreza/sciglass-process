"""Regression test on the real open SciGlass release (skipped unless SCIGLASS_DIR is set).

    git clone --depth 1 https://github.com/drcassar/SciGlass /tmp/SciGlass
    SCIGLASS_DIR=/tmp/SciGlass pytest tests/test_real_data.py
"""
import os

import pytest

SCIGLASS_DIR = os.environ.get("SCIGLASS_DIR")
pytestmark = pytest.mark.skipif(not SCIGLASS_DIR, reason="SCIGLASS_DIR not set")


def test_headline_counts():
    from sciglass_process import coverage, process_by_glass, process_by_record, read_table

    arhiv = read_table(SCIGLASS_DIR, "ARHIV", usecols=["KOD", "INFO", "NUM_GLASS"])
    rec = process_by_record(arhiv)
    assert len(rec) == 39_949
    assert int(rec["has_melt_clause"].sum()) == 27_519
    assert int(rec["has_anneal_clause"].sum()) == 9_608
    # fewer than 1% of melting clauses leave a token unexplained
    melt = rec[rec["has_melt_clause"]]
    assert melt["unparsed"].notna().mean() < 0.01
    scigk = read_table(SCIGLASS_DIR, "SciGK", usecols=["KOD", "GLASNO", "TG"])
    glass = process_by_glass(rec, scigk, properties=("TG",))
    cov = coverage(rec, glass).set_index("item")
    assert cov.loc["glasses with Tg and a melting temperature", "glasses"] > 60_000
