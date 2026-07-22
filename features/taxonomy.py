"""Canonicalizes free-text auction titles into a make/model/generation/trim hierarchy.

Titles are inconsistent across BaT and Cars & Bids ("2006 Mini Cooper S 6-Speed" vs
"MK1 Cooper S, 4-Speed Manual..."), so generation is inferred from (body style keyword,
model year) using each family's known production-year ranges, with explicit chassis-code
mentions (e.g. "E46", "F56") preferred when present. This is inherently heuristic --
good enough for feature engineering, not a VIN decoder.
"""
from __future__ import annotations

import re

YEAR_RE = re.compile(r"(19[5-9]\d|20[0-4]\d)")

MINI_CHASSIS_CODES = re.compile(r"\b(R5[0-9]|F5[4-9]|F60|R6[01])\b", re.IGNORECASE)
MINI_BODY_KEYWORDS = [
    ("clubman", re.compile(r"clubman", re.IGNORECASE)),
    ("countryman", re.compile(r"countryman", re.IGNORECASE)),
    ("paceman", re.compile(r"paceman", re.IGNORECASE)),
    ("convertible", re.compile(r"convertible|cabrio", re.IGNORECASE)),
    ("coupe", re.compile(r"\bcoupe\b", re.IGNORECASE)),
    ("roadster", re.compile(r"\broadster\b", re.IGNORECASE)),
    ("hatch", re.compile(r"hardtop|hatch", re.IGNORECASE)),
]
# (body, year_min, year_max, chassis)
MINI_GEN_TABLE = [
    ("hatch", 2002, 2006, "R50/R53"),
    ("hatch", 2007, 2013, "R56"),
    ("hatch", 2014, 2030, "F55/F56"),
    ("convertible", 2005, 2008, "R52"),
    ("convertible", 2009, 2015, "R57"),
    ("convertible", 2016, 2030, "F57"),
    ("clubman", 2008, 2014, "R55"),
    ("clubman", 2015, 2030, "F54"),
    ("countryman", 2011, 2016, "R60"),
    ("countryman", 2017, 2030, "F60"),
    ("paceman", 2013, 2016, "R61"),
    ("coupe", 2011, 2015, "R58"),
    ("roadster", 2012, 2015, "R59"),
]
MINI_TRIM_KEYWORDS = [
    ("jcw_gp", re.compile(r"john cooper works gp|jcw gp|\bgp3?\b", re.IGNORECASE)),
    ("jcw", re.compile(r"john cooper works|\bjcw\b", re.IGNORECASE)),
    ("cooper_s", re.compile(r"cooper\s*s\b", re.IGNORECASE)),
    ("cooper", re.compile(r"\bcooper\b", re.IGNORECASE)),
    ("one", re.compile(r"\bmini one\b", re.IGNORECASE)),
]

GOLF_GEN_TABLE = [
    (1999, 2005, "Mk4"),
    (2006, 2009, "Mk5"),
    (2010, 2014, "Mk6"),
    (2015, 2021, "Mk7"),
    (2022, 2030, "Mk8"),
]
GOLF_TRIM_KEYWORDS = [
    ("r32", re.compile(r"\br32\b", re.IGNORECASE)),
    ("r", re.compile(r"\bgolf r\b", re.IGNORECASE)),
    ("gti_clubsport", re.compile(r"clubsport|trackspec", re.IGNORECASE)),
    ("gti", re.compile(r"\bgti\b", re.IGNORECASE)),
    ("gli", re.compile(r"\bgli\b", re.IGNORECASE)),
    ("tdi", re.compile(r"\btdi\b", re.IGNORECASE)),
    ("base", re.compile(r".*")),
]

BMW_CHASSIS_CODES = re.compile(r"\b(E30|E34|E36|E39|E46|E61|E91|F31|G21)\b", re.IGNORECASE)
BMW_3_SERIES_RE = re.compile(r"\b3[123][0-9][ei]\b|\bm3\b|\bb3\b|\bb6\b", re.IGNORECASE)
BMW_5_SERIES_RE = re.compile(r"\b5[23][0-9][ei]\b|\bm5\b|\bb10\b", re.IGNORECASE)
# (series, year_min, year_max, chassis)
BMW_GEN_TABLE = [
    ("3", 1996, 1999, "E36"),
    ("3", 1999, 2005, "E46"),
    ("3", 2006, 2012, "E91"),
    ("3", 2013, 2019, "F31"),
    ("3", 2019, 2030, "G21"),
    ("5", 1991, 1996, "E34"),
    ("5", 1997, 2003, "E39"),
    ("5", 2004, 2010, "E61"),
]


def extract_year(text: str) -> int | None:
    m = YEAR_RE.search(text or "")
    return int(m.group(1)) if m else None


def _match_gen_table(table, key, year):
    if year is None:
        return None
    for row in table:
        *keys, ymin, ymax, chassis = row
        if all(k == v for k, v in zip(keys, key)) and ymin <= year <= ymax:
            return chassis
    return None


def classify_mini(title: str, sub_title: str = "") -> dict:
    text = f"{title} {sub_title}"
    year = extract_year(text)

    explicit = MINI_CHASSIS_CODES.search(text)
    body = next((name for name, pat in MINI_BODY_KEYWORDS if pat.search(text)), "hatch")
    chassis = explicit.group(1).upper() if explicit else _match_gen_table(MINI_GEN_TABLE, (body,), year)

    trim = next((name for name, pat in MINI_TRIM_KEYWORDS if pat.search(text)), "unknown")
    gp_edition = bool(re.search(r"\bgp3?\b|john cooper works gp", text, re.IGNORECASE))

    return {
        "make": "MINI",
        "model": "Cooper",
        "body_style": body,
        "generation": chassis,
        "trim": trim,
        "special_edition": gp_edition,
        "year": year,
    }


def classify_golf(title: str, sub_title: str = "") -> dict:
    text = f"{title} {sub_title}"
    year = extract_year(text)
    chassis = None
    if year is not None:
        for ymin, ymax, gen in GOLF_GEN_TABLE:
            if ymin <= year <= ymax:
                chassis = gen
                break

    trim = next((name for name, pat in GOLF_TRIM_KEYWORDS if pat.search(text)), "base")
    special = bool(re.search(r"clubsport|trackspec|rabbit edition|driver'?s edition", text, re.IGNORECASE))

    return {
        "make": "Volkswagen",
        "model": "Golf",
        "body_style": "hatch",
        "generation": chassis,
        "trim": trim,
        "special_edition": special,
        "year": year,
    }


def classify_bmw_wagon(title: str, sub_title: str = "") -> dict:
    text = f"{title} {sub_title}"
    year = extract_year(text)

    explicit = BMW_CHASSIS_CODES.search(text)
    if explicit:
        chassis = explicit.group(1).upper()
        series = "3" if chassis in ("E36", "E46", "E91", "F31", "G21", "E30") else "5"
    else:
        series = "3" if BMW_3_SERIES_RE.search(text) else ("5" if BMW_5_SERIES_RE.search(text) else None)
        chassis = _match_gen_table(BMW_GEN_TABLE, (series,), year) if series else None

    alpina = bool(re.search(r"alpina", text, re.IGNORECASE))
    m_variant = bool(re.search(r"\bm3\b|\bm5\b|competition package", text, re.IGNORECASE))

    return {
        "make": "BMW",
        "model": f"{series}-Series Touring" if series else "Touring",
        "body_style": "wagon",
        "generation": chassis,
        "trim": "alpina" if alpina else ("m" if m_variant else "standard"),
        "special_edition": alpina or m_variant,
        "year": year,
    }


CLASSIFIERS = {
    "mini": classify_mini,
    "vw_golf": classify_golf,
    "bmw_wagon": classify_bmw_wagon,
}


def classify(family: str, title: str, sub_title: str = "") -> dict:
    return CLASSIFIERS[family](title, sub_title)
