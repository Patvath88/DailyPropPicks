import streamlit as st
import requests
from datetime import date, datetime
import pandas as pd

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"  # your ALL STAR key
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Default season (change as needed: 2024 = 2024-25 season)
DEFAULT_SEASON = 2024

# Supported prop types and internal metric keys
PROP_OPTIONS = {
    "Points (PTS)": "pts",
    "Rebounds (REB)": "reb",
    "Assists (AST)": "ast",
    "3PM (FG3M)": "fg3m",
    "Points + Rebounds (PR)": "pr",
    "Points + Assists (PA)": "pa",
    "Rebounds + Assists (RA)": "ra",
    "Points + Rebounds + Assists (PRA)": "pra",
    "Steals (STL)": "stl",
    "Blocks (BLK)": "blk",
    "Steals + Blocks (Stocks)": "stocks",
    "Turnovers (TOV)": "tov",
}


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def fetch(endpoint: str, params: dict | None = None):
    """Generic cached GET wrapper with basic error handling."""
    try:
        resp = requests.get(
            f"{BASE_URL}/{endpoint}",
            headers=HEADERS,
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e), "data": []}


@st.cache_data(show_spinner=False, ttl=300)
def search_players(query: str, per_page: int = 20):
    if not query:
        return []
    data = fetch("players", params={"search": query, "per_page": per_page})
    if "error" in data:
        return []
    return data.get("data", [])


@st.cache_data(show_spinner=False, ttl=300)
def get_season_averages(player_id: int, season: int = DEFAULT_SEASON):
    params = {"season": season, "player_ids[]": player_id}
    data = fetch("season_averages", params=params)
    if "error" in data:
        return None
    arr = data.get("data", [])
    return arr[0] if arr else None


@st.cache_data(show_spinner=False, ttl=300)
def get_recent_stats(player_id: int, season: int, per_page: int = 40):
    """Pull a chunk of regular-season box scores for this player."""
    params = {
        "player_ids[]": player_id,
        "seasons[]": season,
        "per_page": per_page,
        "postseason": "false",
    }
    data = fetch("stats", params=params)
    if "error" in data:
        return []
    return data.get("data", [])


def compute_prop_value_from_row(row: dict, key: str) -> float:
    """Compute a single-game value for the given prop metric key."""
    pts = row.get("pts", 0) or 0
    reb = row.get("reb", 0) or 0
    ast = row.get("ast", 0) or 0
    fg3m = row.get("fg3m", 0) or 0
    stl = row.get("stl", 0) or 0
    blk = row.get("blk", 0) or 0
    tov = row.get("turnover", 0) or 0  # API uses 'turnover'

    if key == "pts":
        return pts
    if key == "reb":
        return reb
    if key == "ast":
        return ast
    if key == "fg3m":
        return fg3m
    if key == "pr":
        return pts + reb
    if key == "pa":
        return pts + ast
    if key == "ra":
        return reb + ast
    if key == "pra":
        return pts + reb + ast
    if key == "stl":
        return stl
    if key == "blk":
        return blk
    if key == "stocks":
        return stl + blk
    if key == "tov":
        return tov

    return 0.0


def compute_average_for_last_n_games(stats: list[dict], key: str, n: int) -> float | None:
    """Average prop value over the last N games."""
    if not stats:
        return None

    # Sort by game date just to be safe
    try:
        sorted_stats = sorted(stats, key=lambda s: s["game"]["date"])
    except Exception:
        sorted_stats = stats

    last_n = sorted_stats[-n:] if len(sorted_stats) >= n else sorted_stats
    if not last_n:
        return None

    vals = [compute_prop_value_from_row(row, key) for row in last_n]
    if not vals:
        return None

    return sum(vals) / len(vals)


@st.cache_data(show_spinner=False, ttl=300)
def get_game_on_date_for_team(team_id: int, game_date: date):
    """Return the game (if any) played by team_id on game_date."""
    date_str = game_date.isoformat()
    params = {
        "team_ids[]": team_id,
        "dates[]": date_str,
        "per_page": 50,
    }
    data = fetch("games", params=params)
    if "error" in data:
        return None

    games = data.get("data", [])
    if not games:
        return None

    # Assume first result is correct for that team/date
    return games[0]


def describe_opponent(game: dict | None, team_id: int) -> str | None:
    if not game:
        return None

    home = game.get("home_team", {})
    away = game.get("visitor_team", {})

    if home.get("id") == team_id:
        opp = away
    else:
        opp = home
    return opp.get("full_name")


# ---------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------
st.set_page_config(
    page_title="NBA Prop Entry Helper",
    layout="centered",
)

st.title("NBA Prop Entry Helper")

st.write(
    "Search a player, pick a game date and prop, and the app will auto-populate "
    "season, last 5, and last 10 averages so you don’t have to type everything in."
)

# Initialize session storage for picks
if "saved_picks" not in st.session_state:
    st.session_state["saved_picks"] = []


# ------------------ PLAYER SEARCH ------------------------
st.subheader("1. Select Player & Game")

col_search, col_season = st.columns([3, 1])

with col_search:
    player_query = st.text_input(
        "Search for player (first or last name)",
        placeholder="e.g. Luka, Curry, Tatum",
    )

with col_season:
    season = st.number_input(
        "Season (YYYY)",
        min_value=2010,
        max_value=2100,
        value=DEFAULT_SEASON,
        step=1,
        help="NBA season year (e.g. 2024 for 2024-25).",
    )

players = search_players(player_query) if player_query else []

selected_player = None
player_label = None

if players:
    options = {
        f"{p['first_name']} {p['last_name']} – {p['team']['full_name']}": p
        for p in players
    }
    player_label = st.selectbox("Choose a player", list(options.keys()))
    selected_player = options[player_label]
elif player_query:
    st.warning("No players found for that search.")

game_date = st.date_input(
    "Game date",
    value=date.today(),
    help="Date of the game for the prop.",
)

prop_label = st.selectbox(
    "Prop stat",
    list(PROP_OPTIONS.keys()),
)
prop_key = PROP_OPTIONS[prop_label]


# ------------------ AUTO STATS FETCH ---------------------
season_avg_val = None
last5_avg = None
last10_avg = None
team_name = None
opp_name = None

if selected_player:
    player_id = selected_player["id"]
    team = selected_player["team"]
    team_name = team.get("full_name", "Unknown team")
    team_id = team["id"]

    st.markdown(f"**Selected:** {player_label}")

    # Season averages
    season_avg = get_season_averages(player_id, season=season)
    if season_avg:
        season_avg_val = compute_prop_value_from_row(season_avg, prop_key)
    else:
        season_avg_val = None

    # Recent stats (box scores)
    stats = get_recent_stats(player_id, season=season, per_page=40)
    last5_avg = compute_average_for_last_n_games(stats, prop_key, n=5)
    last10_avg = compute_average_for_last_n_games(stats, prop_key, n=10)

    # Opponent on date
    game = get_game_on_date_for_team(team_id, game_date)
    opp_name = describe_opponent(game, team_id)

    st.write("### Auto-populated context")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label=f"Season {season} avg ({prop_label})",
            value=f"{season_avg_val:.1f}" if season_avg_val is not None else "N/A",
        )
    with col2:
        st.metric(
            label=f"Last 5 games avg ({prop_label})",
            value=f"{last5_avg:.1f}" if last5_avg is not None else "N/A",
        )
    with col3:
        st.metric(
            label=f"Last 10 games avg ({prop_label})",
            value=f"{last10_avg:.1f}" if last10_avg is not None else "N/A",
        )

    st.write(
        f"**Team:** {team_name}  "
        f"{' | Opponent: ' + opp_name if opp_name else ' | No game found for that date.'}"
    )

    if game:
        st.caption(
            f"Game: {game['visitor_team']['full_name']} at "
            f"{game['home_team']['full_name']} (Status: {game.get('status','')})"
        )


# ------------------ PROP ENTRY FORM ----------------------
st.subheader("2. Enter Your Prop Pick")

with st.form("prop_entry_form"):
    col_line, col_side, col_odds = st.columns(3)

    with col_line:
        prop_line = st.number_input("Prop line", value=0.0, step=0.5)
    with col_side:
        side = st.selectbox("Side", ["Over", "Under"])
    with col_odds:
        odds_str = st.text_input("Odds (e.g. -115, +120)", value="")

    book = st.text_input("Sportsbook (optional)", value="")
    stake = st.number_input(
        "Stake (units or $; optional)",
        min_value=0.0,
        value=0.0,
        step=0.5,
    )
    notes = st.text_area("Notes (optional)", height=80)

    submitted = st.form_submit_button("Save pick")

    if submitted:
        if not selected_player:
            st.error("Select a player first before saving a pick.")
        else:
            pick = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "game_date": game_date.isoformat(),
                "season": season,
                "player_id": selected_player["id"],
                "player_name": f"{selected_player['first_name']} {selected_player['last_name']}",
                "team": team_name,
                "opponent": opp_name,
                "prop": prop_label,
                "prop_key": prop_key,
                "line": prop_line,
                "side": side,
                "odds": odds_str,
                "book": book,
                "stake": stake,
                "season_avg": round(season_avg_val, 2) if season_avg_val is not None else None,
                "last5_avg": round(last5_avg, 2) if last5_avg is not None else None,
                "last10_avg": round(last10_avg, 2) if last10_avg is not None else None,
                "notes": notes,
            }
            st.session_state["saved_picks"].append(pick)
            st.success("Pick saved.")


# ------------------ SAVED PICKS TABLE --------------------
st.subheader("3. Saved Picks")

if st.session_state["saved_picks"]:
    df = pd.DataFrame(st.session_state["saved_picks"])
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download picks as CSV",
        data=csv,
        file_name="nba_prop_picks.csv",
        mime="text/csv",
    )
else:
    st.info("No picks saved yet. Fill out the form and click “Save pick”.")
