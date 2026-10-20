"""Unit tests on hand-written headers that follow the SciGlass ARHIV.INFO grammar."""
import pytest

from sciglass_process.parser import (
    header_of, parse_pairs, parse_process, parse_value, split_clauses, tokenize,
)

H = ("#m 1 pc 2. 1200-1250,a 1 500 3. 1,x 130 1 me 1. old,"
     "r 'Doe J.' 'Roe R.'  'JNCS'  1992 14 2 231#\r\nc m b, cc si na\r\ng 20001, 70 30, 1")


def test_header_is_cut_at_closing_hash():
    assert header_of(H).startswith("m 1 pc") and header_of(H).endswith("231")
    assert header_of("c m b") is None and header_of("") is None


def test_hash_inside_quotes_does_not_end_header():
    h = header_of("#m 1 pc 0 'sample #3 was remelted',r 'X'#\nrest")
    assert "sample #3" in h and h.endswith("'X'")


def test_split_respects_quotes():
    parts = split_clauses("m 1 pc 0 'melted, then cast',a 1 500")
    assert parts == ["m 1 pc 0 'melted, then cast'", "a 1 500"]


def test_simple_melt_and_anneal():
    p = parse_process(H)
    assert p.has_melt_clause and p.has_anneal_clause
    assert p.vessel == "pc"
    assert (p.melt_temp_min_C, p.melt_temp_max_C) == (1200, 1250)
    assert p.melt_time_total_h is None
    assert (p.anneal_temp_min_C, p.anneal_temp_max_C, p.anneal_time_h) == (500, 500, 1)


def test_multi_stage_melt_with_notes():
    p = parse_process("#m 1 clp 2 1500 3. 3 8 cw 0 'and remelting' 2 1500 3 2 0 'then' 2 1600 3. 1#")
    assert [(s.temp_max_C, s.time_h) for s in p.stages] == [(1500, 3), (1500, 2), (1600, 1)]
    assert p.melt_time_total_h == 6 and p.melt_temp_max_C == 1600
    assert p.casting == "cw" and p.notes == ["and remelting", "then"]


def test_minutes_atmosphere_and_synthesis_codes():
    p = parse_process("#m 1 ac 2 1450 3m 30 4 oa 4 st 9 dox,a 1g 20 3m 90 4r 1#")
    assert p.stages[0].time_h == 0.5
    assert p.conditions == ["oa", "st"] and p.bubbling == "dox"
    assert p.anneal_rel_Tg_K == 20 and p.anneal_time_h == 1.5 and p.cooling_rate_K_per_min == 1
    q = parse_process("#m 1s sg 0 'from TEOS'#")
    assert q.synthesis == "sg" and q.stages == [] and q.notes == ["from TEOS"]


def test_cooling_code_and_below_tg():
    p = parse_process("#a 2g 30 3 2 4 slrt#")
    assert p.anneal_rel_Tg_K == -30 and p.cooling_code == "slrt" and p.cooling_rate_K_per_min is None


def test_irregular_tokens_seen_in_the_real_data():
    # quoted note without a "0" marker, a marker glued to its value, unquoted double quotes
    p = parse_process("#m 1 pc 2. 900-1000 3. 0.5 'Pouring in a brass mold.'#")
    assert p.melt_temp_max_C == 1000 and p.stages[0].time_h == 0.5
    assert p.notes == ["Pouring in a brass mold."]
    q = parse_process("#m 1 pc 2 1430 3.6#")
    assert q.stages[0].time_h == 6
    r = parse_process('#m 1 pc 2. 1460  0 "Note. Addition of 1 wt% C."#')
    assert r.notes == ["Note. Addition of 1 wt% C."] and r.unparsed == []


@pytest.mark.parametrize("raw,expected", [
    ("1200-1250", (1200, 1250)), ("1250-1200", (1200, 1250)), ("1450", (1450, 1450)),
    ("1100±20", (1080, 1120)), ("š1000", (1000, 1000)), ("(1000±10)", (990, 1010)),
    ("3.", (3, 3)), ("0,5", (0.5, 0.5)), ("pc", None), ("", None),
])
def test_parse_value(raw, expected):
    assert parse_value(raw) == expected


def test_record_without_header_is_empty():
    p = parse_process("c m b, cc si na\ng 1, 70 30")
    assert not p.has_melt_clause and p.as_row()["n_melt_stages"] == 0


def test_tokenize_keeps_quoted_text_whole():
    assert tokenize("m 0 'a b c' 2 1500") == ["m", "0", "'a b c'", "2", "1500"]
    assert parse_pairs("m", ["0", "'a b'", "2", "1500"]) == [("0", "a b"), ("2", "1500")]


def test_anneal_at_tg_symbol():
    p = parse_process("#a 1 T\u2020 3. 0.5#")
    assert p.anneal_rel_Tg_K == 0 and p.anneal_temp_max_C is None and p.anneal_time_h == 0.5
