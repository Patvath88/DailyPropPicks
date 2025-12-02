import os
import datetime as dt
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"  # your BALLDONTLIE All-Star key
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": API_KEY}

HISTORY_FILE = "prop_history.csv"

st.set_page_config(
    page_title="NBA Prop Research & Entry",
    layout="wide",
)

# -------------------------------------------------------------------
# STYLING (NEW COLORS + BETTER CONTRAST)
# -------------------------------------------------------------------

CUSTOM_CSS = """
<style>
:root {
  --bg-main: #020617;
  --bg-panel: #020617;
  --accent: #6366f1;
  --accent-soft: rgba(99,102,241,0.25);
  --metric-bg: #020617;
  --metric-border: #1e293b;
  --metric-shadow: 0 0 22px rgba(15,23,42,0.9);
  --metric-tag: #9ca3af;
  --metric-text: #f9fafb;
}

body {
  background: radial-gradient(circle at top, #111827 0, #020617 48%, #000000 100%);
  color: #e5e7eb;
}

section.main > div {
  padding-top: 0rem;
}

.block-container {
  padding-top: 1rem;
}

h1, h2, h3, h4, h5 {
  color: #f9fafb;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
  background-color: rgba(15,23,42,0.9);
  border-radius: 999px;
  padding: 0.4rem 0.9rem;
  border: 1px solid rgba(148,163,184,0.35);
}

.stTabs [aria-selected="true"] {
  background: radial-gradient(circle at top left, var(--accent-soft), #020617);
  border-color: var(--accent);
}

.metric-card {
  background: radial-gradient(circle at top left, rgba(99,102,241,0.18), var(--metric-bg));
  border-radius: 0.85rem;
  padding: 0.6rem 0.9rem;
  border: 1px solid var(--metric-border);
  box-shadow: var(--metric-shadow);
}

.metric-tag {
  font-size: 0.72rem;
  text-transform: uppercase;
  color: var(--metric-tag);
  letter-spacing: 0.08em;
}

.metric-value {
  font-size: 1.45rem;
  font-weight: 650;
  color: var(--metric-text);
}

.player-photo-card {
  background: radial-gradient(circle at top, rgba(99,102,241,0.4), #020617);
  border-radius: 1rem;
  padding: 0.35rem;
  border: 1px solid rgba(148,163,184,0.4);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------------
# PROP DEFINITIONS
# -------------------------------------------------------------------

# Map display label -> underlying key logic
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
# HELPERS
# -------------------------------------------------------------------

def fetch_json(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{BASE_URL}/{endpoint}"
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=8)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error on {endpoint}: {e}")
        return {"data": [], "meta": {}}


@st.cache_data(ttl=3600, show_spinner=False)
def get_active_players() -> List[Dict[str, Any]]:
    """Pull all active NBA players (cached)."""
    players = []
    cursor = None
    while True:
        params = {"per_page": 100}
        if cursor is not None:
            params["cursor"] = cursor
        data = fetch_json("players/active", params)
        players.extend(data.get("data", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
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
    cursor = None
    while True:
        params = {
            "player_ids[]": player_id,
            "seasons[]": season,
            "per_page": 100,
        }
        if cursor is not None:
            params["cursor"] = cursor
        data = fetch_json("stats", params)
        stats.extend(data.get("data", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
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
    """Extract numeric value for a given prop key from a stat row."""
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
def get_next_team_game(team_id: int) -> Optional[Dict[str, Any]]:
    today = dt.date.today()
    start = today.strftime("%Y-%m-%d")
    end = (today + dt.timedelta(days=30)).strftime("%Y-%m-%d")
    cursor = None
    best_game = None
    best_date = None

    while True:
        params = {
            "team_ids[]": team_id,
            "start_date": start,
            "end_date": end,
            "per_page": 100,
        }
        if cursor is not None:
            params["cursor"] = cursor
        data = fetch_json("games", params)
        for g in data.get("data", []):
            g_date = dt.datetime.strptime(g["date"], "%Y-%m-%d").date()
            if g_date < today:
                continue
            if best_date is None or g_date < best_date:
                best_date = g_date
                best_game = g
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    return best_game


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
    cursor = None
    while True:
        params = {"team_ids[]": team_id, "per_page": 100}
        if cursor is not None:
            params["cursor"] = cursor
        data = fetch_json("player_injuries", params)
        injuries.extend(data.get("data", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
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
    """
    Use community NBA headshots API (name-based).
    This is best-effort; if the service is down, Streamlit will just fail to load the image.
    """
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
# HISTORY STORAGE
# -------------------------------------------------------------------

def empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
            "H2H Avg vs Opp",
            "Game ID",
            "Actual",
            "Result",
        ]
    )


def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE):
        return empty_history_df()
    try:
        df = pd.read_csv(HISTORY_FILE)
        # make sure all expected columns exist
        missing = [c for c in empty_history_df().columns if c not in df.columns]
        for c in missing:
            df[c] = None
        return df
    except Exception:
        return empty_history_df()


def save_history(df: pd.DataFrame) -> None:
    df.to_csv(HISTORY_FILE, index=False)


def evaluate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Update Actual + Result for any past-dated, ungraded props."""
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
            continue  # game not finished yet

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

tab_form, tab_research, tab_history = st.tabs(
    ["Prop Entry Form", "Player Research", "Prop History"]
)

# -------------------------------------------------------------------
# TAB 1: PROP ENTRY FORM (with delete + headshot)
# -------------------------------------------------------------------

with tab_form:
    st.subheader("Daily Prop Entry")

    col_top1, col_top2 = st.columns([1.2, 1])

    with col_top1:
        game_date = st.date_input(
            "Game date",
            value=dt.date.today(),
            help="Date of the game for this prop.",
        )
        player_label = st.selectbox(
            "Player",
            options=[""] + player_labels,
            index=0,
            help="Search any active NBA player.",
        )

    with col_top2:
        prop_choice = st.selectbox("Prop", options=list(PROP_DEFS.keys()), index=0)
        side_choice = st.selectbox("Side", options=["Over", "Under"], index=0)
        line_value = st.number_input(
            "Prop line (number)",
            step=0.5,
            format="%.2f",
        )
        odds_str = st.text_input("Odds (e.g. -115)", value="")

    if player_label and player_label in player_info:
        info = player_info[player_label]
        player_id = info["id"]
        team_id = info["team_id"]
        team_name = info["team_name"]
        team_abbr = info["team_abbr"]

        img_col, text_col = st.columns([0.9, 2.6])
        with img_col:
            st.markdown('<div class="player-photo-card">', unsafe_allow_html=True)
            st.image(
                get_headshot_url(info["first_name"], info["last_name"]),
                width=110,
                caption="",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with text_col:
            st.markdown(
                f"**{info['first_name']} {info['last_name']}**  \n"
                f"{team_name} ({team_abbr}) · ID `{player_id}`"
            )

        # Matchup info
        game = get_team_game_on_date(team_id, game_date)
        opp_name = "Unknown"
        home_away = "N/A"
        opp_id = None
        game_id = None

        if game:
            opp_team, home_away = get_opponent_from_game(game, team_id)
            opp_name = opp_team["full_name"]
            opp_id = opp_team["id"]
            game_id = game["id"]
            st.markdown(
                f"**Matchup:** {team_abbr} vs {opp_team['abbreviation']} "
                f"({home_away} for {team_abbr}) · Game ID `{game_id}`"
            )
        else:
            st.warning("No scheduled game found for this team on that date.")

        # Stats & averages
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

        h2h_avg = None
        h2h_games = 0
        if opp_id is not None:
            stats_h2h = h2h_stats_vs_team(stats_recent, opp_id)
            h2h_games = len(stats_h2h)
            h2h_avg = average_for_prop(stats_h2h, prop_key)

        st.markdown("##### Recent form for this prop")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        metric_card(mcol1, "Season avg", season_avg)
        metric_card(mcol2, "Last 5 avg", last5_avg)
        metric_card(mcol3, "Last 10 avg", last10_avg)
        metric_card(mcol4, "H2H vs opp avg", h2h_avg, extra=f"{h2h_games} games")

        st.markdown("---")

        if st.button("Add to prop sheet"):
            row = {
                "Date": game_date.strftime("%Y-%m-%d"),
                "Player": player_label,
                "Player ID": player_id,
                "Team": team_name,
                "Opponent": opp_name,
                "Home/Away": home_away,
                "Prop": prop_choice,
                "Side": side_choice,
                "Line": line_value,
                "Odds": odds_str,
                "Season Avg": season_avg,
                "Last 5 Avg": last5_avg,
                "Last 10 Avg": last10_avg,
                "H2H Avg vs Opp": h2h_avg,
                "Game ID": game_id,
                "Actual": None,
                "Result": "Pending",
            }
            st.session_state["prop_rows"].append(row)

            hist_df = load_history()
            hist_df = pd.concat([hist_df, pd.DataFrame([row])], ignore_index=True)
            save_history(hist_df)

            st.success("Prop added to sheet and history.")

    st.markdown("### Current prop sheet")

    if st.session_state["prop_rows"]:
        df_props = pd.DataFrame(st.session_state["prop_rows"])
        df_props_display = df_props.copy()
        df_props_display.index.name = "Row"

        st.dataframe(df_props_display, use_container_width=True, height=400)

        # Delete row control
        del_col1, del_col2 = st.columns([2, 1])
        with del_col1:
            del_idx = st.selectbox(
                "Select row to delete",
                options=df_props_display.index,
                format_func=lambda i: f"{i}: {df_props_display.loc[i, 'Player']} · "
                                      f"{df_props_display.loc[i, 'Prop']} {df_props_display.loc[i, 'Side']} "
                                      f"{df_props_display.loc[i, 'Line']}",
            )
        with del_col2:
            if st.button("Delete selected row"):
                # remove from session
                st.session_state["prop_rows"].pop(int(del_idx))

                # also remove from history file by matching Date/Player/Prop/Line/Odds
                hist_df = load_history()
                mask = ~(
                    (hist_df["Date"] == df_props.loc[del_idx, "Date"])
                    & (hist_df["Player"] == df_props.loc[del_idx, "Player"])
                    & (hist_df["Prop"] == df_props.loc[del_idx, "Prop"])
                    & (hist_df["Line"] == float(df_props.loc[del_idx, "Line"]))
                    & (hist_df["Odds"].astype(str) == str(df_props.loc[del_idx, "Odds"]))
                )
                save_history(hist_df[mask])

                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

        csv = df_props.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="daily_prop_sheet.csv",
            mime="text/csv",
        )
    else:
        st.info("No props added yet. Add a prop above to start building your sheet.")

# -------------------------------------------------------------------
# TAB 2: PLAYER RESEARCH (with headshot)
# -------------------------------------------------------------------

with tab_research:
    st.subheader("Player Research Lab")

    rcol1, rcol2 = st.columns([1.2, 1])

    with rcol1:
        research_player_label = st.selectbox(
            "Player to research",
            options=[""] + player_labels,
            index=0,
        )

    with rcol2:
        research_prop_choice = st.selectbox(
            "Prop focus",
            options=list(PROP_DEFS.keys()),
            index=0,
        )

    if research_player_label and research_player_label in player_info:
        info = player_info[research_player_label]
        player_id = info["id"]
        team_id = info["team_id"]
        team_name = info["team_name"]
        team_abbr = info["team_abbr"]

        img_col, text_col = st.columns([0.9, 2.6])
        with img_col:
            st.markdown('<div class="player-photo-card">', unsafe_allow_html=True)
            st.image(
                get_headshot_url(info["first_name"], info["last_name"]),
                width=130,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with text_col:
            st.markdown(
                f"**{info['first_name']} {info['last_name']}**  \n"
                f"{team_name} ({team_abbr}) · ID `{player_id}`"
            )

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

            next_game = get_next_team_game(team_id)
            opp_team = None
            opp_id = None
            h2h_avg = None
            h2h_games = 0

            if next_game:
                opp_team, home_away = get_opponent_from_game(next_game, team_id)
                opp_id = opp_team["id"]
                st.markdown(
                    f"**Next Game:** {next_game['date']}  ·  "
                    f"{team_abbr} vs {opp_team['abbreviation']} "
                    f"({home_away} for {team_abbr})"
                )

                recent_seasons = [prev_season, current_season]
                stats_for_h2h = get_recent_stats_for_h2h(player_id, recent_seasons)
                stats_h2h = h2h_stats_vs_team(stats_for_h2h, opp_id)
                h2h_games = len(stats_h2h)
                h2h_avg = average_for_prop(stats_h2h, prop_key)
            else:
                st.warning("No upcoming game found for this team in the next 30 days.")

            st.markdown("##### Form snapshot for this prop")
            fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)

            metric_card(
                fcol1,
                "Most recent",
                prop_value(most_recent, prop_key),
                most_recent["game"]["date"],
            )
            metric_card(fcol2, "Last 10 avg", last10_avg)
            metric_card(fcol3, "Last 20 avg", last20_avg)
            metric_card(fcol4, f"{current_season} season avg", season_avg)
            metric_card(fcol5, f"{prev_season} season avg", prev_season_avg)

            if opp_team:
                st.markdown("##### H2H vs next opponent")
                hcol1, hcol2 = st.columns([1, 3])
                metric_card(
                    hcol1,
                    "H2H avg",
                    h2h_avg,
                    extra=f"{h2h_games} games (last 2 seasons)",
                )

                if h2h_games:
                    stats_for_table = sorted(
                        h2h_stats_vs_team(stats_recent, opp_id),
                        key=lambda s: s["game"]["date"],
                        reverse=True,
                    )
                    rows = []
                    for s in stats_for_table:
                        g = s["game"]
                        rows.append(
                            {
                                "Date": g["date"],
                                "Prop Value": prop_value(s, prop_key),
                                "PTS": s.get("pts", 0),
                                "REB": s.get("reb", 0),
                                "AST": s.get("ast", 0),
                                "3PM": s.get("fg3m", 0),
                                "MIN": parse_minutes(s.get("min")),
                            }
                        )
                    df_h2h = pd.DataFrame(rows)
                    hcol2.dataframe(df_h2h, use_container_width=True, height=260)
                else:
                    hcol1.write(
                        "No previous games vs this opponent in the last two seasons."
                    )

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
                    icol2.info(
                        f"No current injuries listed for {opp_team['full_name']}."
                    )

# -------------------------------------------------------------------
# TAB 3: PROP HISTORY (persistent CSV + grading)
# -------------------------------------------------------------------

with tab_history:
    st.subheader("Prop History & Results")

    hist_df = load_history()

    if hist_df.empty:
        st.info("No history saved yet. Add props on the form tab to start tracking.")
    else:
        hist_df["Date_dt"] = pd.to_datetime(hist_df["Date"])
        min_d = hist_df["Date_dt"].min().date()
        max_d = hist_df["Date_dt"].max().date()

        date_range = st.date_input(
            "Filter by date range",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
        )

        if isinstance(date_range, tuple):
            start_d, end_d = date_range
        else:
            start_d = end_d = date_range

        mask = (hist_df["Date_dt"].dt.date >= start_d) & (
            hist_df["Date_dt"].dt.date <= end_d
        )
        view_df = hist_df.loc[mask].drop(columns=["Date_dt"]).sort_values(
            "Date", ascending=False
        )

        st.dataframe(view_df, use_container_width=True, height=500)

        col_u1, col_u2 = st.columns([1, 2])
        with col_u1:
            if st.button("Update results for completed games"):
                updated = evaluate_results(hist_df)
                save_history(updated)
                st.success("Results updated from BallDontLie stats.")
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

        csv_hist = hist_df.drop(columns=["Date_dt"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full history CSV",
            data=csv_hist,
            file_name="prop_history.csv",
            mime="text/csv",
        )
