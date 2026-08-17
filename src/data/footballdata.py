"""Ingest football-data.co.uk into a single normalized match table.

Two file shapes live behind that domain and they do not share a schema:

  main   https://www.football-data.co.uk/mmz4281/{season}/{div}.csv
         One file per division per season. Results + box-score stats + odds.
         Columns grew over time -- 8 in 1993/94, 132 in 2025/26 -- so every
         optional column has to be probed rather than assumed.

  extra  https://www.football-data.co.uk/new/{country}.csv
         One file per country covering all seasons. Closing odds only: no
         pre-close prices, no shots, no corners, no cards, no O/U, no AH.

Verified boundaries (2026-08-15, see docs/research/00-measured-facts.md):
  - Pinnacle closing odds (PSCH/PSCD/PSCA) start at 2012/13. Absent 2011/12.
  - The `Time` column starts at 2019/20. Before that, same-day fixtures cannot
    be ordered, which bounds how rolling features may be built.
  - 22 main divisions carried 7,799 matches in 2023/24.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.football-data.co.uk"
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "footballdata"
OUT_DIR = REPO_ROOT / "data" / "processed"

# The 22 main divisions. Counts in the comment are 2023/24 matches, measured.
MAIN_DIVISIONS: dict[str, str] = {
    "E0": "England Premier League",      # 380
    "E1": "England Championship",        # 552
    "E2": "England League One",          # 552
    "E3": "England League Two",          # 552
    "EC": "England National League",     # 552
    "SC0": "Scotland Premiership",       # 228
    "SC1": "Scotland Championship",      # 180
    "SC2": "Scotland League One",        # 180
    "SC3": "Scotland League Two",        # 180
    "D1": "Germany Bundesliga",          # 306
    "D2": "Germany 2. Bundesliga",       # 306
    "I1": "Italy Serie A",               # 380
    "I2": "Italy Serie B",               # 380
    "SP1": "Spain La Liga",              # 380
    "SP2": "Spain Segunda Division",     # 462
    "F1": "France Ligue 1",              # 306
    "F2": "France Ligue 2",              # 379
    "N1": "Netherlands Eredivisie",      # 306
    "B1": "Belgium Pro League",          # 312
    "P1": "Portugal Primeira Liga",      # 306
    "T1": "Turkey Super Lig",            # 380
    "G1": "Greece Super League",         # 240
}

# Extra-league files. Closing odds only -- see module docstring.
EXTRA_COUNTRIES: list[str] = [
    "ARG", "AUT", "BRA", "CHN", "DNK", "FIN", "IRL", "JPN",
    "MEX", "NOR", "POL", "ROU", "RUS", "SWE", "SWZ", "USA",
]

FIRST_SEASON_START = 1993


def season_codes(first_start: int = FIRST_SEASON_START, last_start: int | None = None) -> list[str]:
    """football-data season codes, e.g. 1993/94 -> '9394', 2023/24 -> '2324'."""
    if last_start is None:
        # A season starting in year Y is published under Y/Y+1. Include the
        # current one; a 404 is how we learn it does not exist yet.
        last_start = pd.Timestamp.today().year
    codes = []
    for start in range(first_start, last_start + 1):
        codes.append(f"{start % 100:02d}{(start + 1) % 100:02d}")
    return codes


# --------------------------------------------------------------------------
# Download + cache
# --------------------------------------------------------------------------

@dataclass
class FetchReport:
    downloaded: int = 0
    cached: int = 0
    missing: int = 0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def _manifest_path() -> Path:
    return RAW_DIR / "_missing.json"


def _load_missing() -> set[str]:
    p = _manifest_path()
    if p.exists():
        return set(json.loads(p.read_text()))
    return set()


def _save_missing(missing: set[str]) -> None:
    # mkdir here rather than relying on the caller: refresh_current() writes the
    # manifest BEFORE download_all() gets a chance to create the tree, so on a
    # cold checkout -- which is every CI run, since data/ is gitignored -- this
    # was a FileNotFoundError before the first request went out.
    _manifest_path().parent.mkdir(parents=True, exist_ok=True)
    _manifest_path().write_text(json.dumps(sorted(missing), indent=0))


def download_all(pause: float = 0.15, session: requests.Session | None = None) -> FetchReport:
    """Download every main division-season and every extra-country file.

    Caches to data/raw/footballdata/. A 404 is recorded in _missing.json so
    later runs skip it -- most division-season combinations genuinely do not
    exist (E0 goes back to 1993/94, G1 does not) and re-probing them every run
    would be thousands of pointless requests.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "main").mkdir(exist_ok=True)
    (RAW_DIR / "extra").mkdir(exist_ok=True)

    sess = session or requests.Session()
    missing = _load_missing()
    rep = FetchReport()

    targets: list[tuple[str, str, Path]] = []
    for code in season_codes():
        for div in MAIN_DIVISIONS:
            key = f"main/{code}/{div}"
            targets.append((key, f"{BASE}/mmz4281/{code}/{div}.csv",
                            RAW_DIR / "main" / f"{code}_{div}.csv"))
    for country in EXTRA_COUNTRIES:
        key = f"extra/{country}"
        targets.append((key, f"{BASE}/new/{country}.csv",
                        RAW_DIR / "extra" / f"{country}.csv"))

    for key, url, dest in targets:
        if dest.exists():
            rep.cached += 1
            continue
        if key in missing:
            rep.missing += 1
            continue
        try:
            r = sess.get(url, timeout=60)
        except requests.RequestException as exc:
            rep.errors.append(f"{key}: {exc}")
            continue
        # 404 is the obvious "no such file". 300 (Multiple Choices) is what
        # this server returns via mod_negotiation for a division-season that
        # never existed -- e.g. G1 in 1993/94. Both mean "not available", and
        # both must be remembered or every later run re-requests them.
        if r.status_code in (300, 404):
            missing.add(key)
            rep.missing += 1
        elif r.status_code == 200 and r.content.strip():
            dest.write_bytes(r.content)
            rep.downloaded += 1
        else:
            rep.errors.append(f"{key}: HTTP {r.status_code}")
        time.sleep(pause)

    _save_missing(missing)
    return rep


def current_season_code(today: pd.Timestamp | None = None) -> str:
    """football-data's season code for the season now under way, e.g. '2627'.

    July belongs to the new season: the cached 2026/27 Scottish Premiership
    file opens on 31/07/2026.
    """
    ts = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    start = ts.year if ts.month >= 7 else ts.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def refresh_current(session: requests.Session | None = None, pause: float = 0.15,
                    today: pd.Timestamp | None = None) -> FetchReport:
    """Re-fetch the current season's files and every extra-country file.

    `download_all` is a cold-start tool and cannot do this. It skips any path
    that already exists and any key memoised in `_missing.json`, with no mtime
    check and no force flag, so on a warm cache the current-season file is
    fetched exactly once and never updated again. A scheduled job relying on it
    would go on predicting forever while no new result ever landed.

    Both caches have to be cleared, and the second one is the trap. Eight of
    the 2026/27 divisions were already memoised missing on 2026-08-17 --
    D1, E1, E2, F1, G1, I1, I2, T1 -- because their files 404 upstream until
    the season starts, while `fixtures.csv` was already carrying E1 and E2
    fixtures. Deleting the files alone leaves those keys in place, and those
    divisions' results never arrive: nothing errors, and the ledger simply
    reports nothing for a third of the corpus.
    """
    code = current_season_code(today)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "main").mkdir(exist_ok=True)
    (RAW_DIR / "extra").mkdir(exist_ok=True)
    missing = _load_missing()
    for div in MAIN_DIVISIONS:
        (RAW_DIR / "main" / f"{code}_{div}.csv").unlink(missing_ok=True)
        missing.discard(f"main/{code}/{div}")
    for country in EXTRA_COUNTRIES:
        # One file per country spanning every season, so it is stale the moment
        # a result lands anywhere in it.
        (RAW_DIR / "extra" / f"{country}.csv").unlink(missing_ok=True)
        missing.discard(f"extra/{country}")
    _save_missing(missing)
    return download_all(pause=pause, session=session)


# --------------------------------------------------------------------------
# Parse
# --------------------------------------------------------------------------

# Columns we keep, mapped to canonical names. Anything absent from a given
# file is filled with NA rather than causing a failure -- schemas vary by era.
MAIN_CORE = {
    "Div": "div", "Date": "date", "Time": "time",
    "HomeTeam": "home_raw", "AwayTeam": "away_raw",
    "FTHG": "fthg", "FTAG": "ftag", "FTR": "ftr",
    "HTHG": "hthg", "HTAG": "htag", "HTR": "htr",
    "Referee": "referee",
    "HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast",
    "HF": "hf", "AF": "af", "HC": "hc", "AC": "ac",
    "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar",
}

# Odds. Suffix C = closing. PS/P = Pinnacle, B365 = Bet365.
MAIN_ODDS = {
    # pre-close 1X2
    "B365H": "b365h", "B365D": "b365d", "B365A": "b365a",
    "PSH": "psh", "PSD": "psd", "PSA": "psa",
    "PH": "psh", "PD": "psd", "PA": "psa",          # older alias for Pinnacle
    "MaxH": "maxh", "MaxD": "maxd", "MaxA": "maxa",
    "AvgH": "avgh", "AvgD": "avgd", "AvgA": "avga",
    # Betfair Exchange pre-close. Added 2026-08-17, and it is load-bearing
    # rather than tidy-up: football-data dropped Pinnacle entirely in 2026/27
    # (the PS*/P* columns are absent from the schema, not empty), so the
    # exchange is now the sharpest price available on both legs. These columns
    # were in the feed all along and were being parsed away.
    "BFEH": "bfeh", "BFED": "bfed", "BFEA": "bfea",
    # closing 1X2  <- the ones that matter
    "B365CH": "b365ch", "B365CD": "b365cd", "B365CA": "b365ca",
    "PSCH": "psch", "PSCD": "pscd", "PSCA": "psca",
    "MaxCH": "maxch", "MaxCD": "maxcd", "MaxCA": "maxca",
    "AvgCH": "avgch", "AvgCD": "avgcd", "AvgCA": "avgca",
    "BFECH": "bfech", "BFECD": "bfecd", "BFECA": "bfeca",
    # totals 2.5
    "B365>2.5": "b365_o25", "B365<2.5": "b365_u25",
    "P>2.5": "ps_o25", "P<2.5": "ps_u25",
    "Avg>2.5": "avg_o25", "Avg<2.5": "avg_u25",
    "B365C>2.5": "b365c_o25", "B365C<2.5": "b365c_u25",
    "PC>2.5": "psc_o25", "PC<2.5": "psc_u25",
    "AvgC>2.5": "avgc_o25", "AvgC<2.5": "avgc_u25",
    "MaxC>2.5": "maxc_o25", "MaxC<2.5": "maxc_u25",
    # asian handicap
    "AHh": "ah_line", "AHCh": "ahc_line",
    "B365AHH": "b365_ahh", "B365AHA": "b365_aha",
    "PAHH": "ps_ahh", "PAHA": "ps_aha",
    "AvgAHH": "avg_ahh", "AvgAHA": "avg_aha",
    "B365CAHH": "b365c_ahh", "B365CAHA": "b365c_aha",
    "PCAHH": "psc_ahh", "PCAHA": "psc_aha",
    "AvgCAHH": "avgc_ahh", "AvgCAHA": "avgc_aha",
    "MaxCAHH": "maxc_ahh", "MaxCAHA": "maxc_aha",
}

EXTRA_MAP = {
    "Country": "country_raw", "League": "league_raw", "Season": "season_raw",
    "Date": "date", "Time": "time",
    "Home": "home_raw", "Away": "away_raw",
    "HG": "fthg", "AG": "ftag", "Res": "ftr",
    "PSCH": "psch", "PSCD": "pscd", "PSCA": "psca",
    "MaxCH": "maxch", "MaxCD": "maxcd", "MaxCA": "maxca",
    "AvgCH": "avgch", "AvgCD": "avgcd", "AvgCA": "avgca",
    "BFECH": "bfech", "BFECD": "bfecd", "BFECA": "bfeca",
    "B365CH": "b365ch", "B365CD": "b365cd", "B365CA": "b365ca",
}

NUMERIC_PREFIXES = ("fthg", "ftag", "hthg", "htag", "hs", "as_", "hst", "ast",
                    "hf", "af", "hc", "ac", "hy", "ay", "hr", "ar")


def _read_csv_bytes(path: Path) -> pd.DataFrame:
    """Read one football-data CSV.

    Three quirks handled here: latin-1 encoding, a UTF-8 BOM on the extra
    files (so the first column reads as 'ï»¿Country'), and trailing all-empty
    rows that pandas otherwise turns into NaN-only records.
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    if "�" in text:  # not valid utf-8 -- fall back
        text = raw.decode("latin-1")
        if text.startswith("﻿"):
            text = text[1:]
    # Some files have ragged trailing commas; csv.reader tolerates it, pandas
    # with the python engine does too.
    df = pd.read_csv(io.StringIO(text), engine="python", on_bad_lines="skip",
                     quoting=csv.QUOTE_MINIMAL)
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    return df


def _parse_dates(s: pd.Series) -> pd.Series:
    """football-data uses dd/mm/yy and dd/mm/yyyy, inconsistently across eras."""
    s = s.astype("string").str.strip()
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    fallback = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return out.fillna(fallback)


def _select(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename the columns we want; leave absent ones absent."""
    present = {src: dst for src, dst in mapping.items() if src in df.columns}
    out = df[list(present)].rename(columns=present)
    # A source column can map to a name already taken (PSH and PH both ->
    # psh). Coalesce rather than carrying duplicates.
    if out.columns.duplicated().any():
        merged = {}
        for name in out.columns.unique():
            block = out.loc[:, out.columns == name]
            merged[name] = block.bfill(axis=1).iloc[:, 0]
        out = pd.DataFrame(merged)
    return out


def load_main_file(path: Path) -> pd.DataFrame | None:
    df = _read_csv_bytes(path)
    if "HomeTeam" not in df.columns or "Date" not in df.columns:
        return None
    keep = _select(df, {**MAIN_CORE, **MAIN_ODDS})
    keep = keep[keep["home_raw"].notna() & keep["date"].notna()].copy()
    if keep.empty:
        return None
    season_code = path.stem.split("_")[0]
    start = int(season_code[:2])
    start += 1900 if start >= 90 else 2000
    keep["season"] = f"{start}-{str(start + 1)[-2:]}"
    keep["source"] = "main"
    if "div" not in keep.columns:
        keep["div"] = path.stem.split("_", 1)[1]
    keep["league"] = keep["div"].map(MAIN_DIVISIONS).fillna(keep["div"])
    return keep


def load_extra_file(path: Path) -> pd.DataFrame | None:
    df = _read_csv_bytes(path)
    if "Home" not in df.columns:
        return None
    keep = _select(df, EXTRA_MAP)
    keep = keep[keep["home_raw"].notna() & keep["date"].notna()].copy()
    if keep.empty:
        return None
    # "2012/2013" -> "2012-13". Single-year seasons (ARG, BRA, USA, JPN) stay.
    def _norm_season(v: str) -> str:
        v = str(v).strip()
        m = re.fullmatch(r"(\d{4})/(\d{4})", v)
        return f"{m.group(1)}-{m.group(2)[-2:]}" if m else v
    keep["season"] = keep["season_raw"].map(_norm_season)
    keep["source"] = "extra"
    keep["div"] = path.stem
    keep["league"] = keep.get("league_raw", pd.Series(path.stem, index=keep.index))
    return keep


# --------------------------------------------------------------------------
# Team identity
# --------------------------------------------------------------------------

# Letters that NFKD does NOT decompose, because they are distinct letters
# rather than a base plus a combining mark. Without this table they survive
# accent-stripping and are then destroyed by the a-z filter: "brondby" was
# coming out as "br ndby", and the same applied to German ss, Polish l and
# Turkish dotless i. Found while joining SportMonks names to football-data.
_TRANSLITERATE = str.maketrans({
    "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe",
    "ß": "ss", "ð": "d", "Ð": "d", "þ": "th", "Þ": "th",
    "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ı": "i", "İ": "i",
    "ǆ": "dz", "ĳ": "ij",
})


def normalize_team(name: str) -> str:
    """Fold a raw team name to a matching key.

    Deliberately conservative: transliterate the non-decomposable letters,
    strip accents, casefold, drop punctuation and common club suffixes,
    collapse whitespace. It does NOT try to unify genuinely different names
    ('AGF' vs 'Aarhus', 'Man United' vs 'Manchester Utd') -- that needs an
    explicit alias map reviewed against real data, because no amount of string
    folding turns one into the other. `team_review()` surfaces candidates.
    """
    s = str(name).translate(_TRANSLITERATE)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold().strip()
    s = re.sub(r"[.'`]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b(fc|afc|cf|sc|ac|as|sv|bk|if|ff|club|kfc|boldklub)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_matches(write: bool = True) -> pd.DataFrame:
    """Parse every cached CSV into one normalized match table."""
    frames: list[pd.DataFrame] = []
    for p in sorted((RAW_DIR / "main").glob("*.csv")):
        f = load_main_file(p)
        if f is not None:
            frames.append(f)
    for p in sorted((RAW_DIR / "extra").glob("*.csv")):
        f = load_extra_file(p)
        if f is not None:
            frames.append(f)
    if not frames:
        raise RuntimeError(f"No parseable CSVs under {RAW_DIR}. Run download_all() first.")

    df = pd.concat(frames, ignore_index=True)

    df["date"] = _parse_dates(df["date"])
    df = df[df["date"].notna()].copy()

    # kickoff = date + time where time exists (2019/20 onward for main files).
    t = df["time"].astype("string").str.strip() if "time" in df.columns else None
    if t is not None:
        td = pd.to_timedelta(t + ":00", errors="coerce")
        df["kickoff"] = df["date"] + td.fillna(pd.Timedelta(0))
        df["has_kickoff_time"] = td.notna()
    else:
        df["kickoff"] = df["date"]
        df["has_kickoff_time"] = False

    for col in NUMERIC_PREFIXES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in df.columns:
        if col.startswith(("b365", "ps", "max", "avg", "bfe", "ah")):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["fthg"].notna() & df["ftag"].notna()].copy()
    df["fthg"] = df["fthg"].astype("int16")
    df["ftag"] = df["ftag"].astype("int16")

    # Derive the result rather than trusting FTR -- it is missing in some
    # early files and mis-cased in others.
    df["result"] = pd.Series(
        pd.NA, index=df.index, dtype="string"
    ).mask(df["fthg"] > df["ftag"], "H").mask(df["fthg"] == df["ftag"], "D").mask(
        df["fthg"] < df["ftag"], "A")

    df["home_key"] = df["home_raw"].map(normalize_team)
    df["away_key"] = df["away_raw"].map(normalize_team)
    df["country"] = df["div"].map(_country_of)

    df = df.sort_values(["kickoff", "div", "home_key"]).reset_index(drop=True)
    df["match_id"] = (
        df["div"].astype(str) + "|" + df["date"].dt.strftime("%Y%m%d")
        + "|" + df["home_key"] + "|" + df["away_key"]
    )
    dupes = df["match_id"].duplicated().sum()
    if dupes:
        df = df[~df["match_id"].duplicated()].copy()

    df.attrs["duplicates_dropped"] = int(dupes)

    # Any column still holding python objects is text. Mixed int/str appears
    # in season_raw, because single-year leagues (ARG, BRA, USA) write 2012
    # while split-year leagues write "2012/2013", and parquet will not infer a
    # type across that.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("string")

    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_DIR / "matches.parquet", index=False)
    return df


def _country_of(div: str) -> str:
    if div.startswith("SC"):
        return "Scotland"
    if div.startswith("E"):
        return "England"
    if div.startswith("D"):
        return "Germany"
    if div.startswith("I"):
        return "Italy"
    if div.startswith("SP"):
        return "Spain"
    if div.startswith("F"):
        return "France"
    if div == "N1":
        return "Netherlands"
    if div == "B1":
        return "Belgium"
    if div == "P1":
        return "Portugal"
    if div == "T1":
        return "Turkey"
    if div == "G1":
        return "Greece"
    return div  # extra files are keyed by country code already


def team_review(df: pd.DataFrame, max_matches: int = 12) -> pd.DataFrame:
    """Surface likely team-name variants for human review.

    Any normalized key appearing in very few matches within a division is
    either a one-season promotion or a spelling variant of a name that is
    already present. This does not decide -- it produces the shortlist.
    """
    long = pd.concat([
        df[["div", "home_key", "season"]].rename(columns={"home_key": "key"}),
        df[["div", "away_key", "season"]].rename(columns={"away_key": "key"}),
    ])
    counts = long.groupby(["div", "key"]).agg(
        matches=("season", "size"), seasons=("season", "nunique")
    ).reset_index()
    return counts[counts["matches"] <= max_matches].sort_values(["div", "matches"])
