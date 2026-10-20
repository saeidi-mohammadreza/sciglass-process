"""Decode the melting (``m``) and annealing (``a``) clauses of SciGlass source records.

Every SciGlass source record (table ``ARHIV``) starts its ``INFO`` field with a
compact header between two ``#`` characters, for example::

    #m 1 pc 2. 1200-1250,a 1 500 3. 1,x 130 1 me 1. old,r 'Hoj J.W.' ...#

The header is a comma-separated list of clauses. The first token of a clause is
its kind (``m`` melting/synthesis, ``a`` annealing, ``x`` property measurement,
``r`` reference). After the kind, ``m`` and ``a`` clauses are a sequence of
*field-number / value* pairs. The field numbers are documented by SciGlass itself
in the ``DT`` table of its property database (``KOD``/``NN``/``SNAME``/``NAME``),
for example ``m 1 pc`` = "Melting in a platinum crucible", ``m 2`` = "at ... °C",
``m 3`` = "for ... h", ``a 1`` = "Annealing at ... °C".

A trailing dot on a field number (``2.``) is the sentence terminator SciGlass uses
when rendering the clause as text; it carries no data. Field ``0`` holds free text.

Published glass-ML pipelines read compositions and properties from the flattened
``SciGK`` table and never parse this header, so the melting temperature, melting
time, crucible, atmosphere, casting and annealing schedule are discarded. This
module turns them into columns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

__all__ = [
    "Clause",
    "MeltStage",
    "Process",
    "header_of",
    "split_clauses",
    "tokenize",
    "parse_pairs",
    "parse_value",
    "parse_process",
    "M_FIELDS",
    "A_FIELDS",
]

# Field-number meaning, from SciGlass's DT table (KOD 'm' and 'a').
M_FIELDS = {
    "0": "note",
    "1": "vessel",          # crucible / furnace code, e.g. pc, sc, ac, clp, ta
    "1s": "synthesis",      # non-melt synthesis: sg (sol-gel), vpd (vapour deposition)
    "1q": "quenching",      # tr (twin roller), bbp (between brass plates)
    "2": "melt_temp_C",
    "2t": "melt_temp_C",
    "3": "melt_time_h",
    "3m": "melt_time_min",
    "4": "conditions",      # atmosphere / stirring / remelting codes
    "8": "casting",         # cw, cmp, cgp, cpmp
    "9": "bubbling",        # da, dox
    "10": "water_method",   # ir
}
A_FIELDS = {
    "0": "note",
    "1": "anneal_temp_C",
    "1g": "anneal_above_Tg_K",
    "2g": "anneal_below_Tg_K",
    "3": "anneal_time_h",
    "3m": "anneal_time_min",
    "4": "cooling",          # a rate in K/min, or a code: sl, slrt
    "4r": "cooling_to_rt_K_per_min",
    "6": "rate_cooling_upper",
    "7": "rate_cooling_lower",
    "10": "anneal_flag",     # st = stabilised samples
}
_FIELD_SETS = {"m": M_FIELDS, "a": A_FIELDS}

_MARKER = re.compile(r"^(\d{1,2}[a-z]?)(\.?)$")
_GLUED = re.compile(r"^(\d{1,2}[a-z]?)\.(\S+)$")  # "3.6" written for "3. 6", "4.oa" for "4. oa"
_TOKEN = re.compile(r"'[^']*'?|\"[^\"]*\"?|[^\s'\"]+")
_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_RANGE = re.compile(rf"^[~<>≈]?\s*({_NUM})\s*(?:-|–|÷|\.\.)\s*({_NUM})$")
_SINGLE = re.compile(rf"^[~<>≈]?\s*({_NUM})$")


@dataclass
class Clause:
    kind: str
    tokens: list[str]
    raw: str


def header_of(info: str | None) -> str | None:
    """Return the text between the leading ``#`` and the closing ``#`` (quotes respected)."""
    if not info:
        return None
    s = info.lstrip()
    if not s.startswith("#"):
        return None
    quote = None
    for i in range(1, len(s)):
        ch = s[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#":
            return s[1:i]
        elif ch in "\r\n":
            return s[1:i]  # unterminated header: stop at end of line
    return s[1:]


def split_clauses(header: str) -> list[str]:
    """Split a header on commas that are not inside quotes."""
    out, cur, quote = [], [], None
    for ch in header:
        if quote:
            if ch == quote:
                quote = None
            cur.append(ch)
        elif ch in "'\"":
            quote = ch
            cur.append(ch)
        elif ch == ",":
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [c.strip() for c in out if c.strip()]


def tokenize(clause: str) -> list[str]:
    return _TOKEN.findall(clause)


def _clauses(info: str | None) -> list[Clause]:
    h = header_of(info)
    if h is None:
        return []
    res = []
    for raw in split_clauses(h):
        toks = tokenize(raw)
        if toks:
            res.append(Clause(toks[0], toks[1:], raw))
    return res


def _unquote(tok: str) -> str:
    if tok[:1] in "'\"":
        tok = tok[1:]
        if tok[-1:] in "'\"":
            tok = tok[:-1]
    return tok.strip()


def parse_pairs(kind: str, tokens: list[str]) -> list[tuple[str, str]]:
    """Turn a clause's tokens into ``(field, value)`` pairs, tolerating the
    irregularities that occur in the real data (a quoted note without its ``0``
    marker, a marker glued to its value, a stray dot)."""
    known = _FIELD_SETS[kind]
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t[:1] in "'\"" or t.startswith(".'"):
            pairs.append(("0", _unquote(t.lstrip("."))))
            i += 1
            continue
        m = _MARKER.match(t)
        if m and m.group(1) in known:
            value = tokens[i + 1] if i + 1 < len(tokens) else ""
            if m.group(1) == "0" or value[:1] in "'\"":
                value = _unquote(value.lstrip("."))
            pairs.append((m.group(1), value.rstrip(".") if m.group(1) != "0" else value))
            i += 2
            continue
        g = _GLUED.match(t)
        if g and g.group(1) in known:
            pairs.append((g.group(1), g.group(2).rstrip(".")))
            i += 1
            continue
        # unexplained token: keep it as a note so nothing is silently lost
        pairs.append(("?", t))
        i += 1
    return pairs


_APPROX = "\u0161~\u2248"  # SciGlass writes "approximately" as \u0161 (cp1252 0x9A)
_TG_SYMBOLS = {"T\u2020", "Tg", "T"}  # SciGlass renders Tg as T\u2020 in this field
_TOL = re.compile(rf"^({_NUM})\s*\u00b1\s*({_NUM})$")


def parse_value(v: str) -> tuple[float, float] | None:
    """``'1200-1250'`` -> (1200, 1250); ``'1450'`` -> (1450, 1450);
    ``'1100\u00b120'`` -> (1080, 1120); ``'\u01611000'`` (approx.) -> (1000, 1000);
    text -> None."""
    v = v.strip().rstrip(".").replace(",", ".")
    if v.startswith("(") and v.endswith(")"):
        v = v[1:-1].strip()
    v = v.lstrip(_APPROX).strip()
    m = _TOL.match(v)
    if m:
        c, t = float(m.group(1)), abs(float(m.group(2)))
        return (c - t, c + t)
    m = _RANGE.match(v)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo, hi) if lo <= hi else (hi, lo)
    m = _SINGLE.match(v)
    if m:
        x = float(m.group(1))
        return (x, x)
    return None


@dataclass
class MeltStage:
    temp_min_C: float | None = None
    temp_max_C: float | None = None
    time_h: float | None = None


@dataclass
class Process:
    """Decoded synthesis history of one SciGlass source record."""

    has_melt_clause: bool = False
    has_anneal_clause: bool = False
    vessel: str | None = None
    synthesis: str | None = None
    quenching: str | None = None
    stages: list[MeltStage] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    casting: str | None = None
    bubbling: str | None = None
    anneal_temp_min_C: float | None = None
    anneal_temp_max_C: float | None = None
    anneal_time_h: float | None = None
    anneal_rel_Tg_K: float | None = None
    cooling_rate_K_per_min: float | None = None
    cooling_code: str | None = None
    notes: list[str] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)
    melt_raw: str | None = None
    anneal_raw: str | None = None

    # ---- summaries used as model features ------------------------------------
    @property
    def melt_temp_max_C(self) -> float | None:
        vals = [s.temp_max_C for s in self.stages if s.temp_max_C is not None]
        return max(vals) if vals else None

    @property
    def melt_temp_min_C(self) -> float | None:
        vals = [s.temp_min_C for s in self.stages if s.temp_min_C is not None]
        return min(vals) if vals else None

    @property
    def melt_time_total_h(self) -> float | None:
        vals = [s.time_h for s in self.stages if s.time_h is not None]
        return sum(vals) if vals else None

    def as_row(self) -> dict:
        return {
            "has_melt_clause": self.has_melt_clause,
            "has_anneal_clause": self.has_anneal_clause,
            "vessel": self.vessel,
            "synthesis": self.synthesis,
            "quenching": self.quenching,
            "n_melt_stages": len(self.stages),
            "melt_temp_min_C": self.melt_temp_min_C,
            "melt_temp_max_C": self.melt_temp_max_C,
            "melt_time_total_h": self.melt_time_total_h,
            "conditions": ";".join(self.conditions) or None,
            "casting": self.casting,
            "bubbling": self.bubbling,
            "anneal_temp_min_C": self.anneal_temp_min_C,
            "anneal_temp_max_C": self.anneal_temp_max_C,
            "anneal_time_h": self.anneal_time_h,
            "anneal_rel_Tg_K": self.anneal_rel_Tg_K,
            "cooling_rate_K_per_min": self.cooling_rate_K_per_min,
            "cooling_code": self.cooling_code,
            "notes": " | ".join(self.notes) or None,
            "unparsed": " ".join(self.unparsed) or None,
            "melt_raw": self.melt_raw,
            "anneal_raw": self.anneal_raw,
        }


def _first_num(v: str) -> float | None:
    r = parse_value(v)
    return r[1] if r else None


def _apply_melt(p: Process, pairs: Iterable[tuple[str, str]]) -> None:
    stage: MeltStage | None = None
    for f, v in pairs:
        if f in ("2", "2t"):
            r = parse_value(v)
            # a new temperature after a completed stage starts the next stage
            if stage is None or stage.temp_max_C is not None:
                stage = MeltStage()
                p.stages.append(stage)
            if r:
                stage.temp_min_C, stage.temp_max_C = r
            else:
                p.unparsed.append(f"{f}:{v}")
        elif f in ("3", "3m"):
            x = _first_num(v)
            if x is None:
                p.unparsed.append(f"{f}:{v}")
                continue
            if f == "3m":
                x /= 60.0
            if stage is None or stage.time_h is not None:
                stage = MeltStage()
                p.stages.append(stage)
            stage.time_h = x
        elif f == "1":
            p.vessel = p.vessel or v
        elif f == "1s":
            p.synthesis = v
        elif f == "1q":
            p.quenching = v
        elif f == "4":
            p.conditions.append(v)
        elif f == "8":
            p.casting = v
        elif f == "9":
            p.bubbling = v
        elif f == "0":
            if v:
                p.notes.append(v)
        elif f == "10":
            pass
        else:
            p.unparsed.append(f"{f}:{v}" if f != "?" else v)


def _apply_anneal(p: Process, pairs: Iterable[tuple[str, str]]) -> None:
    for f, v in pairs:
        if f == "1":
            r = parse_value(v)
            if not r and v.lstrip(_APPROX) in _TG_SYMBOLS:
                p.anneal_rel_Tg_K = 0.0  # "annealing at Tg"
                continue
            if r and p.anneal_temp_max_C is None:
                p.anneal_temp_min_C, p.anneal_temp_max_C = r
            elif not r:
                p.unparsed.append(f"a{f}:{v}")
        elif f in ("1g", "2g"):
            x = _first_num(v)
            if x is not None:
                p.anneal_rel_Tg_K = x if f == "1g" else -x
        elif f in ("3", "3m"):
            x = _first_num(v)
            if x is not None:
                p.anneal_time_h = x / 60.0 if f == "3m" else x
        elif f in ("4", "4r"):
            x = _first_num(v)
            if x is not None:
                p.cooling_rate_K_per_min = x
            else:
                p.cooling_code = v
        elif f == "0":
            if v:
                p.notes.append(v)
        elif f in ("6", "7", "10"):
            pass
        else:
            p.unparsed.append(f"a{f}:{v}" if f != "?" else v)


def parse_process(info: str | None) -> Process:
    """Decode the melting and annealing clauses of one ``ARHIV.INFO`` value."""
    p = Process()
    for c in _clauses(info):
        if c.kind == "m":
            p.has_melt_clause = True
            p.melt_raw = c.raw if p.melt_raw is None else p.melt_raw + " , " + c.raw
            _apply_melt(p, parse_pairs("m", c.tokens))
        elif c.kind == "a":
            p.has_anneal_clause = True
            p.anneal_raw = c.raw if p.anneal_raw is None else p.anneal_raw + " , " + c.raw
            _apply_anneal(p, parse_pairs("a", c.tokens))
    return p
