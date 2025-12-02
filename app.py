import datetime as dt
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# Your ALL-STAR BallDontLie key (NBA)
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"  # replace if needed
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": API_KEY}

st.set_page_config(
    page_title="NBA Prop Research & Entry",
    layout="wide",
)

# -------------------------------------------------------------------
# BASIC STYLING
# -------------------------------------------------------------------

CUSTOM_CSS = """
<style>
body {
    background-color: #050711;
}
section.main > div {
    padding-top: 0rem;
}
.block-container {
    padding-top: 1rem;
}
h1, h2, h3, h4, h5 {
    color: #f5f5ff;
}
.metric-tag {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #aaa;
}
.metric-value {
    font-size: 1.3rem;
    font-weight: 600;
}
.metric-card {
    background: #101322;
    border-radius: 0.75rem;
    padding: 0.6rem 0.8rem;
    border: 1px solid #262a3f;
}
.prop-table .stDataFrame {
    border-radius: 0.75rem;
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
# HELPERS
# -------------------------------------------------------------------

def fetch_json(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generic GET wrapper with basic error handling."""
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
    """NBA season code: year of season start (e.g. 2024 for 2024-25)."""
    if today is None:
        today = dt.date.today()
    if today.month >= 10:
        return today.year
    return today.year - 1


@st.cache_data(ttl=900, show_spinner=False)
def get_player_stats_for_season(player_id: int, season: int) -> List[Dict[str, Any]]:
    """All regular-season stats for a player in a given season."""
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
    """Return the game for a team on a specific date, if any."""
    date_str = date.strftime("%Y-%m-%d")
    params = {
        "team_ids[]": team_id,
        "dates[]": date_str,
        "per_page": 100,
    }
    data = fetch_json("games", params)
    games = data.get("data", [])
    if not games:
        return None
    # If multiple (rare), pick first
    return games[0]


@st.cache_data(ttl=900, show_spinner=False)
def get_next_team_game(team_id: int) -> Optional[Dict[str, Any]]:
    """Next upcoming game for a team from today forward."""
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
    """Return opponent team dict and 'Home'/'Away' flag for player team."""
    home = game["home_team"]
    visitor = game["visitor_team"]
    if home["id"] == team_id:
        return visitor, "Home"
    else:
        return home, "Away"


@st.cache_data(ttl=900, show_spinner=False)
def get_recent_stats_for_h2h(player_id: int, seasons: List[int]) -> List[Dict[str, Any]]:
    """Stats across multiple seasons for H2H and recent form."""
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


# -------------------------------------------------------------------
# MAIN UI
# -------------------------------------------------------------------

if "prop_rows" not in st.session_state:
    st.session_state["prop_rows"] = []

st.title("NBA Prop Research & Entry")

players = get_active_players()
if not players:
    st.stop()

player_labels, player_info = build_player_index(players)

tab_form, tab_research = st.tabs(["Prop Entry Form", "Player Research"])


# -------------------------------------------------------------------
# TAB 1: PROP ENTRY FORM
# -------------------------------------------------------------------

with tab_form:
    st.subheader("Daily Prop Entry")

    col_top1, col_top2 = st.columns([1, 1])

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
        prop_choice = st.selectbox(
            "Prop",
            options=list(PROP_DEFS.keys()),
            index=0,
        )
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

        st.markdown(
            f"**Team:** {team_name} ({team_abbr})  |  **Player ID:** `{player_id}`"
        )

        # Determine opponent from schedule for that date
        game = get_team_game_on_date(team_id, game_date)
        opp_name = "Unknown"
        home_away = "N/A"
        opp_id = None

        if game:
            opp_team, home_away = get_opponent_from_game(game, team_id)
            opp_name = opp_team["full_name"]
            opp_id = opp_team["id"]
            st.markdown(
                f"**Matchup:** {team_abbr} vs {opp_team['abbreviation']}  "
                f"({home_away} for {team_abbr})"
            )
        else:
            st.warning("No scheduled game found for this team on that date.")

        # Stats & averages
        today = dt.date.today()
        current_season = get_current_season(today)
        prev_season = current_season - 1

        stats_current = get_player_stats_for_season(player_id, current_season)
        stats_prev = get_player_stats_for_season(player_id, prev_season)
        stats_recent = sorted(
            stats_prev + stats_current, key=lambda s: s["game"]["date"]
        )

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

        # Metric cards
        st.markdown("##### Recent form for this prop")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)

        def metric(col, label, val, extra=""):
            with col:
                st.markdown(
                    f"""<div class="metric-card">
<span class="metric-tag">{label}</span><br>
<span class="metric-value">{'—' if val is None else val}</span>
<div style="font-size:0.7rem;color:#888;margin-top:2px;">{extra}</div>
</div>""",
                    unsafe_allow_html=True,
                )

        metric(mcol1, "Season avg", season_avg)
        metric(mcol2, "Last 5 avg", last5_avg)
        metric(mcol3, "Last 10 avg", last10_avg)
        extra = f"{h2h_games} games" if h2h_games else ""
        metric(mcol4, "H2H vs opp avg", h2h_avg, extra)

        st.markdown("---")

        if st.button("Add to prop sheet"):
            row = {
                "Date": game_date.strftime("%Y-%m-%d"),
                "Player": player_label,
                "Team": team_name,
                "Opponent": opp_name,
                "Home/Away": home_away,
                "Prop": prop_choice,
                "Line": line_value,
                "Odds": odds_str,
                "Season Avg": season_avg,
                "Last 5 Avg": last5_avg,
                "Last 10 Avg": last10_avg,
                "H2H Avg vs Opp": h2h_avg,
            }
            st.session_state["prop_rows"].append(row)
            st.success("Prop added to sheet.")

    st.markdown("### Current prop sheet")
    if st.session_state["prop_rows"]:
        df_props = pd.DataFrame(st.session_state["prop_rows"])
        st.dataframe(df_props, use_container_width=True, height=400)
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
# TAB 2: PLAYER RESEARCH
# -------------------------------------------------------------------

with tab_research:
    st.subheader("Player Research Lab")

    rcol1, rcol2 = st.columns([1, 1])

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

        st.markdown(
            f"**Team:** {team_name} ({team_abbr})  |  **Player ID:** `{player_id}`"
        )

        today = dt.date.today()
        current_season = get_current_season(today)
        prev_season = current_season - 1

        prop_key = PROP_DEFS[research_prop_choice]

        # Get stats
        stats_current = get_player_stats_for_season(player_id, current_season)
        stats_prev = get_player_stats_for_season(player_id, prev_season)
        stats_recent = sorted(
            stats_prev + stats_current,
            key=lambda s: s["game"]["date"],
        )

        if not stats_recent:
            st.warning("No stats found for this player yet.")
            st.stop()

        # Recent performance & averages
        most_recent = get_most_recent_stat(stats_recent)
        last10_avg = last_n_average(stats_recent, prop_key, 10)
        last20_avg = last_n_average(stats_recent, prop_key, 20)
        season_avg = average_for_prop(stats_current, prop_key)
        prev_season_avg = average_for_prop(stats_prev, prop_key)

        # Next upcoming game for H2H + injuries
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

        def metric2(col, label, val, extra=""):
            with col:
                st.markdown(
                    f"""<div class="metric-card">
<span class="metric-tag">{label}</span><br>
<span class="metric-value">{'—' if val is None else val}</span>
<div style="font-size:0.7rem;color:#888;margin-top:2px;">{extra}</div>
</div>""",
                    unsafe_allow_html=True,
                )

        metric2(
            fcol1,
            "Most recent",
            prop_value(most_recent, prop_key),
            most_recent["game"]["date"],
        )
        metric2(fcol2, "Last 10 avg", last10_avg)
        metric2(fcol3, "Last 20 avg", last20_avg)
        metric2(fcol4, f"{current_season} season avg", season_avg)
        metric2(fcol5, f"{prev_season} season avg", prev_season_avg)

        if opp_team:
            st.markdown("##### H2H vs next opponent")
            hcol1, hcol2 = st.columns([1, 3])
            metric2(
                hcol1,
                "H2H avg",
                h2h_avg,
                extra=f"{h2h_games} games (last 2 seasons)",
            )

            # Mini game-log table vs opponent
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
                hcol1.write("No previous games vs this opponent in last two seasons.")

        # Injuries
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
