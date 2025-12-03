# app.py — Mobile-first NBA Prop Research & Entry (automatic matchup & minutes)
import os
import math
import uuid
import statistics
import datetime as dt
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"  # BallDontLie ALL-STAR key
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": API_KEY}

HISTORY_FILE = "prop_history.csv"
PARLAY_FILE = "parlay_history.csv"

st.set_page_config(
    page_title="NBA Prop Research & Entry",
    layout="wide",
)

# -------------------------------------------------------------------
# MOBILE-FIRST STYLING
# -------------------------------------------------------------------

CUSTOM_CSS = """
<style>
:root{
  --bg:#020617;
  --panel:#071026;
  --accent:#6366f1;
  --muted:#9ca3af;
  --text:#f8fafc;
  --card:#0b1220;
  --card-border: rgba(148,163,184,0.08);
}

/* Page & container */
body {
  background: linear-gradient(180deg, #071021 0%, #020617 80%);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
  -webkit-font-smoothing: antialiased;
}

/* Make the main block a little denser on mobile */
.block-container {
  padding-top: 0.75rem;
  padding-left: 0.6rem;
  padding-right: 0.6rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; margin-bottom: 0.6rem; }
.stTabs [data-baseweb="tab"] {
  padding: 0.36rem 0.8rem; border-radius: 999px; font-size: 0.92rem;
  background: rgba(255,255,255,0.02); border: 1px solid rgba(148,163,184,0.04);
}

/* Metric cards - bigger and higher contrast on mobile */
.metric-card {
  background: linear-gradient(180deg, rgba(99,102,241,0.12), rgba(2,6,23,0.7));
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 10px;
  border: 1px solid var(--card-border);
}
.metric-tag { font-size: 0.72rem; color: var(--muted); letter-spacing: 0.06em; }
.metric-value { font-size: 1.4rem; font-weight: 700; color: var(--text); margin-top: 6px; }

/* Player photo card scaled for phone */
.player-photo-card { border-radius: 12px; padding: 6px; display:inline-block; margin-bottom:6px; }
.player-photo-card img { border-radius: 8px; width: 110px; height: auto; object-fit: cover; }

/* Buttons full width on mobile for easy tap */
div.stButton > button, button[kind] {
  min-height:44px !important;
  padding: 10px 14px !important;
  font-size: 15px !important;
}

/* Inputs full width and with more spacing */
input, select, textarea {
  font-size: 15px !important;
  padding: 10px !important;
}

/* Dataframe container to avoid tiny fonts on mobile */
.stDataFrame table {
  font-size: 13px;
}

/* Expanders: make them touch-friendly */
details > summary {
  padding: 10px 8px;
  border-radius: 8px;
  background: rgba(255,255,255,0.01);
  margin-bottom: 8px;
}

/* Responsive tweaks via media queries */
@media (max-width: 800px) {
  .metric-card { padding: 14px; }
  .metric-value { font-size: 1.6rem; }
  .player-photo-card img { width: 96px; }
  .block-container { padding-left: 8px; padding-right: 8px; }
}

/* Desktop keep look */
@media (min-width: 1200px) {
  .player-photo-card img { width: 140px; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------------
# PROP DEFINITIONS
# -------------------------------------------------------------------

PROP_DEFS = {
    "Points": "pts",
    "Rebounds": "reb",
    "Assists": "ast",
    "3PT Made": "fg3m",
    "Points + Rebounds (PR)": "pts+reb",
    "Points + Assists (PA)": "pts+ast",
    "Rebounds + Assists (RA)": "reb+ast",
    "Points + Rebounds + Assists (PRA)": "pts+reb+ast",
    "Steals": "stl",
    "Blocks": "blk",
    "Steals + Blocks": "stl+blk",
    "Turnovers": "tov",
    "Minutes": "min",
}

# -------------------------------------------------------------------
# GENERIC HELPERS
# -------------------------------------------------------------------

def fetch_json(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{BASE_URL}/{endpoint}"
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error on {endpoint}: {e}")
        return {"data": [], "meta": {}}


@st.cache_data(ttl=3600, show_spinner=False)
def get_active_players() -> List[Dict[str, Any]]:
    players = []
    page = 1
    while True:
        params = {"per_page": 100, "page": page}
        data = fetch_json("players", params)
        items = data.get("data", [])
        players.extend(items)
        if len(items) < 100:
            break
        page += 1
    # filter active players if available (some tiers may include active flag)
    return players


def build_player_index(players: List[Dict[str, Any]]):
    labels = []
    info = {}
    for p in players:
        team = p.get("team") or {}
        label = f"{p['first_name']} {p['last_name']} ({team.get('abbreviation', 'FA')})"
        labels.append(label)
        info[label] = {
            "id": p["id"],
            "first_name": p["first_name"],
            "last_name": p["last_name"],
            "team_id": team.get("id"),
            "team_name": team.get("full_name"),
            "team_abbr": team.get("abbreviation"),
        }
    labels = sorted(labels)
    return labels, info


def get_current_season(today: Optional[dt.date] = None) -> int:
    if today is None:
        today = dt.date.today()
    return today.year if today.month >= 10 else today.year - 1


@st.cache_data(ttl=900, show_spinner=False)
def get_player_stats_for_season(player_id: int, season: int) -> List[Dict[str, Any]]:
    stats = []
    page = 1
    while True:
        params = {
            "player_ids[]": player_id,
            "seasons[]": season,
            "per_page": 100,
            "page": page,
        }
        data = fetch_json("stats", params)
        batch = data.get("data", [])
        stats.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return stats


def parse_minutes(min_str: Any) -> float:
    if not min_str:
        return 0.0
    s = str(min_str)
    if ":" in s:
        m, sec = s.split(":")
        return int(m) + int(sec) / 60.0
    try:
        return float(s)
    except Exception:
        return 0.0


def prop_value(stat: Dict[str, Any], key: str) -> float:
    pts = stat.get("pts", 0) or 0
    reb = stat.get("reb", 0) or 0
    ast = stat.get("ast", 0) or 0
    stl = stat.get("stl", 0) or 0
    blk = stat.get("blk", 0) or 0
    tov = stat.get("turnover", 0) or 0
    fg3m = stat.get("fg3m", 0) or 0

    if key == "pts":
        return pts
    if key == "reb":
        return reb
    if key == "ast":
        return ast
    if key == "stl":
        return stl
    if key == "blk":
        return blk
    if key == "stl+blk":
        return stl + blk
    if key == "tov":
        return tov
    if key == "fg3m":
        return fg3m
    if key == "pts+reb":
        return pts + reb
    if key == "pts+ast":
        return pts + ast
    if key == "reb+ast":
        return reb + ast
    if key == "pts+reb+ast":
        return pts + reb + ast
    if key == "min":
        return parse_minutes(stat.get("min"))
    return 0.0


def average_for_prop(stats: List[Dict[str, Any]], key: str) -> Optional[float]:
    if not stats:
        return None
    vals = [prop_value(s, key) for s in stats]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 1)


def last_n_average(stats: List[Dict[str, Any]], key: str, n: int) -> Optional[float]:
    if not stats:
        return None
    stats_sorted = sorted(stats, key=lambda s: s["game"]["date"])
    slice_ = stats_sorted[-n:]
    return average_for_prop(slice_, key)


def get_most_recent_stat(stats: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not stats:
        return None
    return max(stats, key=lambda s: s["game"]["date"])


@st.cache_data(ttl=900, show_spinner=False)
def get_team_game_on_date(team_id: int, date: dt.date) -> Optional[Dict[str, Any]]:
    date_str = date.strftime("%Y-%m-%d")
    params = {"team_ids[]": team_id, "dates[]": date_str, "per_page": 100}
    data = fetch_json("games", params)
    games = data.get("data", [])
    return games[0] if games else None


@st.cache_data(ttl=900, show_spinner=False)
def get_team_games_for_season(team_id: int, season: int) -> List[Dict[str, Any]]:
    games = []
    page = 1
    while True:
        params = {
            "team_ids[]": team_id,
            "seasons[]": season,
            "per_page": 100,
            "page": page,
        }
        data = fetch_json("games", params)
        batch = data.get("data", [])
        games.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return games


def get_opponent_from_game(game: Dict[str, Any], team_id: int) -> (Dict[str, Any], str):
    home = game["home_team"]
    visitor = game["visitor_team"]
    if home["id"] == team_id:
        return visitor, "Home"
    else:
        return home, "Away"


@st.cache_data(ttl=900, show_spinner=False)
def get_recent_stats_for_h2h(player_id: int, seasons: List[int]) -> List[Dict[str, Any]]:
    stats_all: List[Dict[str, Any]] = []
    for season in seasons:
        stats_all.extend(get_player_stats_for_season(player_id, season))
    return stats_all


def h2h_stats_vs_team(stats: List[Dict[str, Any]], opp_team_id: int) -> List[Dict[str, Any]]:
    return [
        s for s in stats
        if s["game"]["home_team_id"] == opp_team_id
        or s["game"]["visitor_team_id"] == opp_team_id
    ]


@st.cache_data(ttl=300, show_spinner=False)
def get_team_injuries(team_id: int) -> List[Dict[str, Any]]:
    injuries = []
    page = 1
    while True:
        params = {"team_ids[]": team_id, "per_page": 100, "page": page}
        data = fetch_json("player_injuries", params)
        batch = data.get("data", [])
        injuries.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return injuries


def injuries_to_df(injuries: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for inj in injuries:
        p = inj.get("player", {})
        rows.append(
            {
                "Player": f"{p.get('first_name', '')} {p.get('last_name', '')}",
                "Status": inj.get("status"),
                "Return": inj.get("return_date"),
                "Note": inj.get("description"),
            }
        )
    return pd.DataFrame(rows)


def get_headshot_url(first: str, last: str) -> str:
    def slug(s: str) -> str:
        s = s.lower()
        for ch in [" ", ".", "'", "`"]:
            s = s.replace(ch, "_")
        return s
    return f"https://nba-players.herokuapp.com/players/{slug(last)}/{slug(first)}"


def metric_card(col, label: str, value: Optional[float], extra: str = ""):
    display_val = "—" if value is None else value
    col.markdown(
        f"""<div class="metric-card">
<span class="metric-tag">{label}</span><br>
<span class="metric-value">{display_val}</span>
<div style="font-size:0.7rem;color:#9ca3af;margin-top:2px;">{extra}</div>
</div>""",
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------
# TEAM DEFENSIVE SCORE & MINUTES PREDICTION HELPERS
# -------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_team_points_allowed_avg(team_id: int, season: int) -> Optional[float]:
    """
    Compute average points allowed per game for the team in the season
    by inspecting the game scores from BallDontLie.
    """
    games = get_team_games_for_season(team_id, season)
    if not games:
        return None
    allowed = []
    for g in games:
        try:
            if g["home_team"]["id"] == team_id:
                opp_score = g.get("visitor_team_score", None)
            else:
                opp_score = g.get("home_team_score", None)
            if opp_score is not None:
                allowed.append(opp_score)
        except Exception:
            continue
    if not allowed:
        return None
    return float(sum(allowed) / len(allowed))


def map_points_allowed_to_def_score(points_allowed: Optional[float]) -> float:
    """
    Map points allowed per game to 0.1-0.9 scale where lower = tougher defense.
    Uses 110 as central league average and a spread of ~20 points for scaling.
    """
    if points_allowed is None:
        return 0.5
    league_avg = 110.0
    diff = points_allowed - league_avg  # positive => softer defense
    # scale diff (-inf..inf) -> offset [-0.4,0.4] approx
    offset = (diff / 20.0) * 0.4
    raw = 0.5 + offset
    return float(max(0.1, min(0.9, raw)))


def predict_player_minutes(player_stats_recent: List[Dict[str, Any]], season_stats: List[Dict[str, Any]], team_id: Optional[int]) -> (Optional[float], float):
    """
    Return (predicted_minutes, minutes_adj) where minutes_adj is in [-1,1].
    Strategy:
      - Use last 10 games avg minutes if available, else season avg minutes.
      - Inspect team injuries and bump minutes by ~8% per injured rotation player (cap 25%).
      - Derive minutes_adj by comparing predicted minutes to season avg minutes scaled by ~12 minutes.
    """
    season_min_avg = average_for_prop(season_stats, "min") or 0.0
    last10_min = last_n_average(player_stats_recent, "min", 10) or season_min_avg or 0.0

    bump = 0.0
    if team_id is not None:
        try:
            inj = get_team_injuries(team_id)
            # count likely rotation-impact injuries (status not None and includes Out or Questionable)
            count_inj = 0
            for x in inj:
                st_text = (x.get("status") or "").lower()
                if st_text and ("out" in st_text or "questionable" in st_text or "day_to_day" in st_text or "doubtful" in st_text):
                    count_inj += 1
            bump = min(0.25, 0.08 * count_inj)  # 8% per injury, cap 25%
        except Exception:
            bump = 0.0

    predicted_minutes = last10_min * (1.0 + bump)
    # fallback if still zero
    if predicted_minutes <= 0:
        predicted_minutes = season_min_avg

    # convert to minutes_adj (-1..1)
    denom = 12.0
    minutes_adj = (predicted_minutes - (season_min_avg or predicted_minutes)) / denom if denom else 0.0
    minutes_adj = max(-1.0, min(1.0, minutes_adj))
    return round(predicted_minutes, 1), minutes_adj

# -------------------------------------------------------------------
# PROBABILITY / GRADING HELPERS
# -------------------------------------------------------------------

def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def implied_prob_from_american(odds: Optional[float]) -> float:
    if odds is None:
        return 0.535
    if odds < 0:
        return -odds / (-odds + 100.0)
    else:
        return 100.0 / (odds + 100.0)


def american_to_decimal(odds: float) -> float:
    if odds < 0:
        return 1 + 100.0 / -odds
    else:
        return 1 + odds / 100.0


def parse_american_odds(raw: str) -> Optional[float]:
    try:
        s = str(raw).strip()
        s = s.replace("+", "")
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def grade_from_prob_edge(p_hit: Optional[float], edge: Optional[float]) -> str:
    if p_hit is None or edge is None:
        return "N/A"

    if p_hit >= 0.65 and edge >= 0.12:
        return "A+"
    if p_hit >= 0.60 and edge >= 0.08:
        return "A"
    if p_hit >= 0.57 and edge >= 0.05:
        return "B+"
    if p_hit >= 0.55 and edge >= 0.02:
        return "B"
    if -0.02 <= edge <= 0.02:
        return "C"
    if edge < -0.06 and p_hit < 0.52:
        return "F"
    return "D"


def compute_expected_and_grade(
    values_recent: List[float],
    l5: Optional[float],
    l10: Optional[float],
    l20: Optional[float],
    season: Optional[float],
    line: float,
    side: str,
    opp_def_score: float,
    minutes_adj: float,
    odds_float: Optional[float],
) -> (Optional[float], Optional[float], Optional[float], str):
    if not values_recent or season is None:
        return None, None, None, "N/A"

    def safe(v, fallback):
        return fallback if v is None else v

    season_val = season
    l10_val = safe(l10, season_val)
    l5_val = safe(l5, l10_val)
    l20_val = safe(l20, season_val)

    base_form = (
        0.45 * l10_val +
        0.15 * l5_val +
        0.10 * l20_val +
        0.30 * season_val
    )

    matchup_adj = 1.0 + (opp_def_score - 0.5) * 0.16  # ±8%
    injury_adj = 1.0 + minutes_adj * 0.15  # ±15%

    expected_stat = base_form * matchup_adj * injury_adj

    if len(values_recent) > 1:
        std = statistics.pstdev(values_recent)
    else:
        std = max(abs(values_recent[0]) / 3.0, 0.5)
    std = max(std, 0.5)

    if side == "Over":
        z = (line - expected_stat) / std
        p_hit = 1.0 - normal_cdf(z)
    else:
        z = (line - expected_stat) / std
        p_hit = normal_cdf(z)

    p_hit = max(0.01, min(0.99, p_hit))

    book_p = implied_prob_from_american(odds_float)
    edge = p_hit - book_p
    grade = grade_from_prob_edge(p_hit, edge)

    return round(expected_stat, 2), round(p_hit, 3), round(edge, 3), grade

# -------------------------------------------------------------------
# BACKFILL HELPER
# -------------------------------------------------------------------

def recompute_missing_model_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows with missing ExpectedStat/HitProb/Edge/Grade, recompute model metrics.
    Only backfills recent rows (last 7 days) to avoid heavy API calls.
    Uses neutral matchup/minutes for backfill to be safe.
    """
    cutoff = dt.date.today() - dt.timedelta(days=7)

    for idx, row in df.iterrows():
        val = row.get("ExpectedStat")
        if not (val is None or str(val).lower() in ("none", "nan", "")):
            continue

        try:
            d = dt.datetime.strptime(str(row["Date"]), "%Y-%m-%d").date()
        except Exception:
            continue
        if d < cutoff:
            continue

        try:
            player_id = int(row["Player ID"])
        except Exception:
            continue

        prop_label = str(row["Prop"])
        prop_key = PROP_DEFS.get(prop_label)
        if not prop_key:
            continue

        try:
            line = float(row["Line"])
        except Exception:
            continue

        side = str(row.get("Side", "Over"))
        odds_float = parse_american_odds(row.get("Odds", ""))

        season = get_current_season(d)
        prev_season = season - 1

        stats_current = get_player_stats_for_season(player_id, season)
        stats_prev = get_player_stats_for_season(player_id, prev_season)
        stats_recent = sorted(stats_prev + stats_current, key=lambda s: s["game"]["date"])

        if not stats_recent:
            continue

        season_avg = average_for_prop(stats_current, prop_key)
        last5_avg = last_n_average(stats_recent, prop_key, 5)
        last10_avg = last_n_average(stats_recent, prop_key, 10)
        last20_avg = last_n_average(stats_recent, prop_key, 20)

        values_recent = [
            prop_value(s, prop_key)
            for s in (stats_recent[-20:] if len(stats_recent) > 20 else stats_recent)
        ]

        if not values_recent or season_avg is None:
            continue

        # safe neutral values for backfill
        expected_stat, hit_prob, edge, grade = compute_expected_and_grade(
            values_recent,
            last5_avg,
            last10_avg,
            last20_avg,
            season_avg,
            line,
            side,
            opp_def_score=0.5,
            minutes_adj=0.0,
            odds_float=odds_float,
        )

        df.at[idx, "Season Avg"] = season_avg
        df.at[idx, "Last 5 Avg"] = last5_avg
        df.at[idx, "Last 10 Avg"] = last10_avg
        df.at[idx, "Last 20 Avg"] = last20_avg
        df.at[idx, "ExpectedStat"] = expected_stat
        df.at[idx, "HitProb"] = hit_prob
        df.at[idx, "Edge"] = edge
        df.at[idx, "Grade"] = grade

    return df

# -------------------------------------------------------------------
# HISTORY STORAGE
# -------------------------------------------------------------------

def empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "RowID",
            "Date",
            "Player",
            "Player ID",
            "Team",
            "Opponent",
            "Home/Away",
            "Prop",
            "Side",
            "Line",
            "Odds",
            "Season Avg",
            "Last 5 Avg",
            "Last 10 Avg",
            "Last 20 Avg",
            "H2H Avg vs Opp",
            "Game ID",
            "ExpectedStat",
            "HitProb",
            "Edge",
            "Grade",
            "Actual",
            "Result",
        ]
    )


def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE):
        return empty_history_df()
    try:
        df = pd.read_csv(HISTORY_FILE)
        template_cols = list(empty_history_df().columns)
        for col in template_cols:
            if col not in df.columns:
                df[col] = None
        return df[template_cols]
    except Exception:
        return empty_history_df()


def save_history(df: pd.DataFrame) -> None:
    df.to_csv(HISTORY_FILE, index=False)


def evaluate_results(df: pd.DataFrame) -> pd.DataFrame:
    today = dt.date.today()
    for idx, row in df.iterrows():
        result = str(row.get("Result", ""))
        if result in ("Hit", "Miss", "Push"):
            continue

        try:
            game_date = dt.datetime.strptime(str(row["Date"]), "%Y-%m-%d").date()
        except Exception:
            continue

        if game_date >= today:
            continue

        try:
            player_id = int(row["Player ID"])
            game_id = int(row["Game ID"])
        except Exception:
            continue

        prop_label = str(row["Prop"])
        prop_key = PROP_DEFS.get(prop_label)
        if not prop_key:
            continue

        stats_data = fetch_json(
            "stats",
            {
                "player_ids[]": player_id,
                "game_ids[]": game_id,
                "per_page": 10,
            },
        ).get("data", [])

        if not stats_data:
            df.at[idx, "Result"] = "No data"
            continue

        stat = stats_data[0]
        actual = prop_value(stat, prop_key)
        df.at[idx, "Actual"] = actual

        try:
            line_val = float(row["Line"])
        except Exception:
            df.at[idx, "Result"] = "Line error"
            continue

        side = str(row.get("Side", "Over"))
        if abs(actual - line_val) < 1e-6:
            res = "Push"
        elif side == "Over":
            res = "Hit" if actual > line_val else "Miss"
        else:
            res = "Hit" if actual < line_val else "Miss"

        df.at[idx, "Result"] = res

    return df

# -------------------------------------------------------------------
# PARLAY STORAGE
# -------------------------------------------------------------------

def empty_parlay_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ParlayID",
            "Name",
            "Date",
            "LegRowIDs",
            "Stake",
            "BookOdds",
            "ParlayProb",
            "BookImpliedProb",
            "Edge",
            "EV_per_unit",
        ]
    )


def load_parlay_history() -> pd.DataFrame:
    if not os.path.exists(PARLAY_FILE):
        return empty_parlay_df()
    try:
        df = pd.read_csv(PARLAY_FILE)
        template_cols = list(empty_parlay_df().columns)
        for col in template_cols:
            if col not in df.columns:
                df[col] = None
        return df[template_cols]
    except Exception:
        return empty_parlay_df()


def save_parlay_history(df: pd.DataFrame) -> None:
    df.to_csv(PARLAY_FILE, index=False)

# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------

if "prop_rows" not in st.session_state:
    st.session_state["prop_rows"] = []

# -------------------------------------------------------------------
# MAIN UI
# -------------------------------------------------------------------

st.title("NBA Prop Research & Entry")

players = get_active_players()
if not players:
    st.stop()

player_labels, player_info = build_player_index(players)

tab_form, tab_research, tab_history, tab_parlay = st.tabs(
    ["Prop Entry Form", "Player Research", "Prop History", "Parlay Builder"]
)

# ---------------------------
# TAB 1: Prop Entry (mobile form) - automatic matchup & minutes
# ---------------------------
with tab_form:
    st.subheader("Daily Prop Entry")

    # Mobile-first form
    with st.form(key="prop_entry_form", clear_on_submit=False):
        game_date = st.date_input("Game date", value=dt.date.today())
        player_search = st.text_input("Search player (2+ letters)", placeholder="Type first or last name")
        player_options = [""]  # default blank
        if player_search and len(player_search.strip()) >= 2:
            matching = [lab for lab in player_labels if player_search.strip().lower() in lab.lower()]
            player_options = [""] + matching
        player_label = st.selectbox("Player", options=player_options, index=0)

        if player_label and player_label in player_info:
            info = player_info[player_label]
            st.markdown(f"<div style='display:flex;align-items:center;gap:12px'>", unsafe_allow_html=True)
            st.markdown(f"<div class='player-photo-card'><img src='{get_headshot_url(info['first_name'], info['last_name'])}'/></div>", unsafe_allow_html=True)
            st.markdown(f"<div><b>{info['first_name']} {info['last_name']}</b><br/>{info['team_name']} ({info['team_abbr']})</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        prop_choice = st.selectbox("Prop", options=list(PROP_DEFS.keys()), index=0)
        side_choice = st.selectbox("Side", options=["Over", "Under"], index=0)
        line_value = st.number_input("Line (number)", step=0.5, format="%.2f")
        odds_str = st.text_input("Odds (optional, e.g. -115)", value="")

        st.markdown("Model snapshot will compute when you press Add to sheet. Opponent defense & minutes are calculated automatically.")

        col_btn_left, col_btn_right = st.columns([1,1])
        with col_btn_left:
            add_pressed = st.form_submit_button("➕ Add to prop sheet")
        with col_btn_right:
            clear_pressed = st.form_submit_button("🧹 Clear inputs")

    # Handle form actions
    if 'add_pressed' in locals() and add_pressed:
        if not (player_label and player_label in player_info):
            st.warning("Pick a player first.")
        else:
            info = player_info[player_label]
            player_id = info["id"]
            team_id = info["team_id"]
            team_name = info["team_name"]
            team_abbr = info["team_abbr"]

            # find the scheduled game and opponent
            game = get_team_game_on_date(team_id, game_date)
            opp_name = "Unknown"; home_away="N/A"; opp_id=None; game_id=None
            if game:
                opp_team, home_away = get_opponent_from_game(game, team_id)
                opp_name = opp_team["full_name"]; opp_id = opp_team["id"]; game_id = game["id"]

            today = dt.date.today()
            current_season = get_current_season(today)
            prev_season = current_season - 1

            stats_current = get_player_stats_for_season(player_id, current_season)
            stats_prev = get_player_stats_for_season(player_id, prev_season)
            stats_recent = sorted(stats_prev + stats_current, key=lambda s: s["game"]["date"])

            prop_key = PROP_DEFS[prop_choice]
            season_avg = average_for_prop(stats_current, prop_key)
            last5_avg = last_n_average(stats_recent, prop_key, 5)
            last10_avg = last_n_average(stats_recent, prop_key, 10)
            last20_avg = last_n_average(stats_recent, prop_key, 20)

            # automatic opponent defensive score
            opp_def_score = 0.5
            if opp_id is not None:
                pts_allowed = get_team_points_allowed_avg(opp_id, current_season)
                opp_def_score = map_points_allowed_to_def_score(pts_allowed)

            # predicted minutes & minutes_adj
            predicted_minutes, minutes_adj = predict_player_minutes(stats_recent, stats_current, team_id)

            # compute expected & grade
            values_recent = [
                prop_value(s, prop_key)
                for s in (stats_recent[-20:] if len(stats_recent) > 20 else stats_recent)
            ]
            odds_float = parse_american_odds(odds_str)

            if values_recent and season_avg is not None:
                expected_stat, hit_prob, edge, grade = compute_expected_and_grade(
                    values_recent,
                    last5_avg,
                    last10_avg,
                    last20_avg,
                    season_avg,
                    float(line_value),
                    side_choice,
                    opp_def_score,
                    minutes_adj,
                    odds_float,
                )
            else:
                expected_stat = hit_prob = edge = None
                grade = "N/A"

            row_id = str(uuid.uuid4())
            row = {
                "RowID": row_id,
                "Date": game_date.strftime("%Y-%m-%d"),
                "Player": player_label,
                "Player ID": player_id,
                "Team": team_name,
                "Opponent": opp_name,
                "Home/Away": home_away,
                "Prop": prop_choice,
                "Side": side_choice,
                "Line": float(line_value),
                "Odds": odds_str,
                "Season Avg": season_avg,
                "Last 5 Avg": last5_avg,
                "Last 10 Avg": last10_avg,
                "Last 20 Avg": last20_avg,
                "H2H Avg vs Opp": None,
                "Game ID": game_id,
                "ExpectedStat": expected_stat,
                "HitProb": hit_prob,
                "Edge": edge,
                "Grade": grade,
                "Actual": None,
                "Result": "Pending",
            }

            st.session_state["prop_rows"].append(row)
            hist_df = load_history()
            hist_df = pd.concat([hist_df, pd.DataFrame([row])], ignore_index=True)
            save_history(hist_df)
            st.success("Prop added to sheet and prop history.")

    if 'clear_pressed' in locals() and clear_pressed:
        st.experimental_rerun()

    # Show current sheet (compact)
    st.markdown("### Current prop sheet (tap a row to view)")
    if st.session_state["prop_rows"]:
        df_props = pd.DataFrame(st.session_state["prop_rows"])
        # Ensure RowID exists
        if "RowID" not in df_props.columns:
            df_props["RowID"] = [str(uuid.uuid4()) for _ in range(len(df_props))]
            for i, rid in df_props["RowID"].items():
                st.session_state["prop_rows"][i]["RowID"] = rid
        else:
            for i in df_props.index:
                if pd.isna(df_props.at[i, "RowID"]) or df_props.at[i, "RowID"] == "":
                    new_id = str(uuid.uuid4())
                    df_props.at[i, "RowID"] = new_id
                    st.session_state["prop_rows"][i]["RowID"] = new_id

        df_display = df_props.copy()
        df_display.index.name = "Row"
        st.dataframe(df_display[
            ["Player","Prop","Side","Line","Odds","ExpectedStat","HitProb","Grade","Result"]
        ], use_container_width=True, height=320)

        # Delete
        del_col1, del_col2 = st.columns([2,1])
        with del_col1:
            del_idx = st.selectbox(
                "Select row to delete",
                options=df_display.index,
                format_func=lambda i: f"{i}: {df_display.loc[i,'Player']} · {df_display.loc[i,'Prop']} {df_display.loc[i,'Line']}"
            )
        with del_col2:
            if st.button("Delete selected row"):
                row_to_delete = df_props.loc[del_idx]
                st.session_state["prop_rows"].pop(int(del_idx))
                hist_df = load_history()
                rid = row_to_delete.get("RowID")
                if rid and "RowID" in hist_df.columns:
                    hist_df = hist_df[hist_df["RowID"] != rid]
                    save_history(hist_df)
                st.experimental_rerun()

        # Group by game (expanders)
        st.markdown("### Grouped by game")
        df_group = df_props.copy()
        df_group["GameLabel"] = df_group.apply(
            lambda r: f"{r['Date']} · {r['Team']} vs {r['Opponent']} ({r['Home/Away']})",
            axis=1,
        )
        for game_label, gdf in df_group.groupby("GameLabel"):
            with st.expander(game_label, expanded=False):
                st.dataframe(
                    gdf[["Player","Prop","Side","Line","Odds","ExpectedStat","HitProb","Grade"]],
                    use_container_width=True,
                )
    else:
        st.info("No props added yet.")

# ---------------------------
# TAB 2: Player Research
# ---------------------------
with tab_research:
    st.subheader("Player Research Lab")

    rcol1, rcol2 = st.columns([1.2, 1])
    with rcol1:
        research_player_label = st.selectbox("Player to research", options=[""] + player_labels, index=0)
    with rcol2:
        research_prop_choice = st.selectbox("Prop focus", options=list(PROP_DEFS.keys()), index=0)

    if research_player_label and research_player_label in player_info:
        info = player_info[research_player_label]
        player_id = info["id"]
        team_id = info["team_id"]
        team_name = info["team_name"]
        team_abbr = info["team_abbr"]

        img_col, text_col = st.columns([0.9, 2.6])
        with img_col:
            st.markdown('<div class="player-photo-card">', unsafe_allow_html=True)
            st.image(get_headshot_url(info["first_name"], info["last_name"]), width=130)
            st.markdown("</div>", unsafe_allow_html=True)

        with text_col:
            st.markdown(f"**{info['first_name']} {info['last_name']}**  \n{team_name} ({team_abbr}) · ID `{player_id}`")

        today = dt.date.today()
        current_season = get_current_season(today)
        prev_season = current_season - 1

        prop_key = PROP_DEFS[research_prop_choice]

        stats_current = get_player_stats_for_season(player_id, current_season)
        stats_prev = get_player_stats_for_season(player_id, prev_season)
        stats_recent = sorted(stats_prev + stats_current, key=lambda s: s["game"]["date"])

        if not stats_recent:
            st.warning("No stats found for this player yet.")
        else:
            most_recent = get_most_recent_stat(stats_recent)
            last10_avg = last_n_average(stats_recent, prop_key, 10)
            last20_avg = last_n_average(stats_recent, prop_key, 20)
            season_avg = average_for_prop(stats_current, prop_key)
            prev_season_avg = average_for_prop(stats_prev, prop_key)

            next_game = get_team_game_on_date(team_id, dt.date.today())
            opp_team = None
            opp_id = None
            h2h_avg = None
            h2h_games = 0

            if next_game:
                opp_team, home_away = get_opponent_from_game(next_game, team_id)
                opp_id = opp_team["id"]
                st.markdown(f"**Next Game:** {next_game['date']}  ·  {team_abbr} vs {opp_team['abbreviation']} ({home_away} for {team_abbr})")
                recent_seasons = [prev_season, current_season]
                stats_for_h2h = get_recent_stats_for_h2h(player_id, recent_seasons)
                stats_h2h = h2h_stats_vs_team(stats_for_h2h, opp_id)
                h2h_games = len(stats_h2h)
                h2h_avg = average_for_prop(stats_h2h, prop_key)
            else:
                st.warning("No upcoming game found for this team in the next 30 days.")

            st.markdown("##### Form snapshot for this prop")
            fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)
            metric_card(fcol1, "Most recent", prop_value(most_recent, prop_key), most_recent["game"]["date"])
            metric_card(fcol2, "Last 10 avg", last10_avg)
            metric_card(fcol3, "Last 20 avg", last20_avg)
            metric_card(fcol4, f"{current_season} season avg", season_avg)
            metric_card(fcol5, f"{prev_season} season avg", prev_season_avg)

            if opp_team:
                st.markdown("##### H2H vs next opponent")
                hcol1, hcol2 = st.columns([1, 3])
                metric_card(hcol1, "H2H avg", h2h_avg, extra=f"{h2h_games} games (last 2 seasons)")
                if h2h_games:
                    stats_for_table = sorted(h2h_stats_vs_team(stats_recent, opp_id), key=lambda s: s["game"]["date"], reverse=True)
                    rows = []
                    for s in stats_for_table:
                        g = s["game"]
                        rows.append({
                            "Date": g["date"],
                            "Prop Value": prop_value(s, prop_key),
                            "PTS": s.get("pts", 0),
                            "REB": s.get("reb", 0),
                            "AST": s.get("ast", 0),
                            "3PM": s.get("fg3m", 0),
                            "MIN": parse_minutes(s.get("min")),
                        })
                    df_h2h = pd.DataFrame(rows)
                    hcol2.dataframe(df_h2h, use_container_width=True, height=260)
                else:
                    hcol1.write("No previous games vs this opponent in the last two seasons.")

            st.markdown("##### Injury report (may impact minutes/usage)")
            icol1, icol2 = st.columns(2)
            team_injuries = get_team_injuries(team_id)
            if team_injuries:
                df_team_inj = injuries_to_df(team_injuries)
                icol1.markdown(f"**{team_name} injuries**")
                icol1.dataframe(df_team_inj, use_container_width=True, height=230)
            else:
                icol1.info(f"No current injuries listed for {team_name}.")
            if opp_team:
                opp_injuries = get_team_injuries(opp_team["id"])
                if opp_injuries:
                    df_opp_inj = injuries_to_df(opp_injuries)
                    icol2.markdown(f"**{opp_team['full_name']} injuries**")
                    icol2.dataframe(df_opp_inj, use_container_width=True, height=230)
                else:
                    icol2.info(f"No current injuries listed for {opp_team['full_name']}.")

# ---------------------------
# TAB 3: Prop History
# ---------------------------
with tab_history:
    st.subheader("Prop History & Results")

    hist_df = load_history()
    hist_df = recompute_missing_model_metrics(hist_df)
    save_history(hist_df)

    if hist_df.empty:
        st.info("No history saved yet. Add props on the form tab to start tracking.")
    else:
        hist_df["Date_dt"] = pd.to_datetime(hist_df["Date"], errors="coerce")
        min_d = hist_df["Date_dt"].min().date()
        max_d = hist_df["Date_dt"].max().date()
        date_range = st.date_input("Filter by date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        if isinstance(date_range, tuple):
            start_d, end_d = date_range
        else:
            start_d = end_d = date_range
        mask = (hist_df["Date_dt"].dt.date >= start_d) & (hist_df["Date_dt"].dt.date <= end_d)
        view_df = hist_df.loc[mask].drop(columns=["Date_dt"]).sort_values("Date", ascending=False)
        st.dataframe(view_df, use_container_width=True, height=420)

        col_u1, col_u2 = st.columns([1,2])
        with col_u1:
            if st.button("Update results for completed games"):
                updated = evaluate_results(hist_df)
                save_history(updated)
                st.success("Results updated from BallDontLie stats.")
                st.experimental_rerun()
        csv_hist = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download full history CSV", data=csv_hist, file_name="prop_history.csv", mime="text/csv")

# ---------------------------
# TAB 4: Parlay Builder (no parlay grade)
# ---------------------------
with tab_parlay:
    st.subheader("Parlay Builder")

    if not st.session_state["prop_rows"]:
        st.info("No props in the current sheet. Add props on the form tab first.")
    else:
        df_props = pd.DataFrame(st.session_state["prop_rows"])
        # Ensure RowIDs
        if "RowID" not in df_props.columns:
            df_props["RowID"] = [str(uuid.uuid4()) for _ in range(len(df_props))]
            for i, rid in df_props["RowID"].items():
                st.session_state["prop_rows"][i]["RowID"] = rid
        else:
            for i in df_props.index:
                if pd.isna(df_props.at[i, "RowID"]) or df_props.at[i, "RowID"] == "":
                    new_id = str(uuid.uuid4())
                    df_props.at[i, "RowID"] = new_id
                    st.session_state["prop_rows"][i]["RowID"] = new_id

        df_props.index.name = "Row"
        leg_indices = st.multiselect(
            "Select legs",
            options=df_props.index.tolist(),
            format_func=lambda i: f"{i}: {df_props.loc[i,'Player']} · {df_props.loc[i,'Prop']} {df_props.loc[i,'Side']} {df_props.loc[i,'Line']} ({df_props.loc[i,'Odds']})"
        )

        pcol1, pcol2, pcol3 = st.columns([1,1,1])
        with pcol1:
            parlay_name = st.text_input("Parlay name", value="")
        with pcol2:
            stake = st.number_input("Stake (units)", min_value=0.0, value=1.0, step=0.5)
        with pcol3:
            parlay_odds_str = st.text_input("Book parlay odds (e.g. +600)", value="")

        parlay_prob = ev_per_unit = parlay_edge = None

        if leg_indices:
            leg_probs = []
            for i in leg_indices:
                p = df_props.loc[i].get("HitProb")
                if p is None or pd.isna(p):
                    p = 0.5
                leg_probs.append(float(p))

            parlay_prob = 1.0
            for p in leg_probs:
                parlay_prob *= p
            parlay_prob = max(0.0001, min(0.999, parlay_prob))

            parlay_odds = parse_american_odds(parlay_odds_str)
            book_p = implied_prob_from_american(parlay_odds)

            if parlay_odds is not None:
                dec = american_to_decimal(parlay_odds)
                ev_per_unit = parlay_prob * (dec - 1.0) - (1.0 - parlay_prob)
            else:
                ev_per_unit = None

            parlay_edge = parlay_prob - book_p
            st.markdown("##### Parlay summary")
            p_mcol1, p_mcol2, p_mcol3 = st.columns(3)
            metric_card(p_mcol1, "Model parlay prob", None if parlay_prob is None else f"{parlay_prob*100:.2f}%")
            metric_card(p_mcol2, "Book implied prob", None if book_p is None else f"{book_p*100:.2f}%")
            metric_card(p_mcol3, "Parlay EV (per unit)", None if ev_per_unit is None else round(ev_per_unit,3))

            if st.button("Save parlay"):
                parlay_id = str(uuid.uuid4())
                leg_row_ids = ";".join([str(df_props.loc[i].get("RowID")) for i in leg_indices])
                row = {
                    "ParlayID": parlay_id,
                    "Name": parlay_name,
                    "Date": dt.date.today().strftime("%Y-%m-%d"),
                    "LegRowIDs": leg_row_ids,
                    "Stake": stake,
                    "BookOdds": parlay_odds_str,
                    "ParlayProb": parlay_prob,
                    "BookImpliedProb": book_p,
                    "Edge": parlay_edge,
                    "EV_per_unit": ev_per_unit,
                }
                p_hist = load_parlay_history()
                p_hist = pd.concat([p_hist, pd.DataFrame([row])], ignore_index=True)
                save_parlay_history(p_hist)
                st.success("Parlay saved.")

        st.markdown("### Parlay history")
        p_hist = load_parlay_history()
        if p_hist.empty:
            st.info("No parlays saved yet.")
        else:
            st.dataframe(p_hist, use_container_width=True, height=240)
            csv_parlay = p_hist.to_csv(index=False).encode("utf-8")
            st.download_button("Download parlay history CSV", data=csv_parlay, file_name="parlay_history.csv", mime="text/csv")

            st.markdown("### Parlay legs (expanded view)")
            hist_df = load_history()
            for _, prow in p_hist.iterrows():
                label = f"{prow['Date']} · {prow.get('Name') or prow['ParlayID']}"
                with st.expander(label):
                    leg_ids_str = prow.get("LegRowIDs")
                    if pd.isna(leg_ids_str) or str(leg_ids_str).strip() == "":
                        st.write("No leg IDs stored.")
                        continue
                    leg_ids = [x for x in str(leg_ids_str).split(";") if x and x.lower() != "nan"]
                    if not leg_ids:
                        st.write("No valid leg IDs stored.")
                        continue
                    df_legs = hist_df[hist_df["RowID"].astype(str).isin(leg_ids)]
                    if df_legs.empty:
                        st.write("No matching props found in history for these legs.")
                    else:
                        st.dataframe(df_legs[[
                            "Date","Player","Team","Opponent","Prop","Side","Line","Odds","ExpectedStat","HitProb","Grade","Result"
                        ]], use_container_width=True, height=260)

# End of file
