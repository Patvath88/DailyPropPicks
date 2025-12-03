# app.py
# Mobile-first NBA Prop Research & Entry
# - Automatic minutes model (trained in-app using scikit-learn RandomForest)
# - Opponent defense cache (daily)
# - Predicted minutes + OppDefScore saved to history and shown in tables
# - Mobile share pick card (dark theme)
# - Robust headshot fallback (tries public endpoints then silhouette)
# - On-demand player search (rate-limit safe)

import os
import math
import uuid
import time
import json
import base64
import statistics
import datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"  # replace if needed
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": API_KEY}

HISTORY_FILE = "prop_history.csv"
PARLAY_FILE = "parlay_history.csv"
PLAYERS_CACHE_FILE = Path("players_cache.json")
PLAYERS_CACHE_TTL = 24 * 3600
DEF_RATINGS_FILE = Path("def_ratings.json")
MODEL_FILE = Path("minutes_model.joblib")
MODEL_META_FILE = Path("minutes_model_meta.json")

st.set_page_config(page_title="NBA Prop Research & Entry", layout="wide")

# -----------------------------
# MOBILE CSS
# -----------------------------
CUSTOM_CSS = """
<style>
:root{
  --bg:#05060a; --muted:#9ca3af; --text:#f8fafc;
  --card:#0b1220; --card-border: rgba(148,163,184,0.06);
}
.block-container { padding-top:0.75rem; padding-left:0.6rem; padding-right:0.6rem; }
.metric-card { background: linear-gradient(180deg, rgba(99,102,241,0.08), rgba(2,6,23,0.7)); border-radius:12px; padding:10px; margin-bottom:8px; border:1px solid var(--card-border); }
.metric-tag{ font-size:0.72rem; color:var(--muted); } .metric-value{ font-size:1.4rem; font-weight:700; color:var(--text); }
.player-photo-card img { border-radius:8px; width:110px; height:auto; object-fit:cover; }
div.stButton > button { min-height:44px !important; font-size:15px !important; padding:10px 14px !important; }
.stDataFrame table { font-size:13px; }
@media (max-width:800px) { .player-photo-card img{width:96px;} .metric-value{font-size:1.6rem;} }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# PROP DEFINITIONS
# -----------------------------
PROP_DEFS = {
    "Points": "pts", "Rebounds": "reb", "Assists": "ast", "3PT Made": "fg3m",
    "Points + Rebounds (PR)": "pts+reb", "Points + Assists (PA)": "pts+ast",
    "Rebounds + Assists (RA)": "reb+ast", "Points + Rebounds + Assists (PRA)": "pts+reb+ast",
    "Steals": "stl", "Blocks": "blk", "Steals + Blocks": "stl+blk", "Turnovers": "tov", "Minutes": "min"
}

# -----------------------------
# UTIL: API fetch + small backoff
# -----------------------------
def fetch_json(endpoint: str, params: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
    url = f"{BASE_URL}/{endpoint}"
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as he:
        # Surface 429 cleanly
        if resp is not None and resp.status_code == 429:
            st.warning(f"API 429 Too Many Requests for {endpoint}. Try again in a moment.")
        else:
            st.error(f"API error {getattr(resp,'status_code', '')} on {endpoint}: {he}")
        return {"data": [], "meta": {}}
    except Exception as e:
        st.error(f"API fetch error on {endpoint}: {e}")
        return {"data": [], "meta": {}}

# -----------------------------
# PLAYERS: cache + search (no heavy paging)
# -----------------------------
def _read_players_cache() -> List[Dict[str,Any]]:
    try:
        if not PLAYERS_CACHE_FILE.exists(): return []
        payload = json.loads(PLAYERS_CACHE_FILE.read_text())
        ts = payload.get("ts",0)
        if time.time() - ts > PLAYERS_CACHE_TTL:
            return []
        return payload.get("players",[])
    except Exception:
        return []

def _write_players_cache(players: List[Dict[str,Any]]):
    try:
        PLAYERS_CACHE_FILE.write_text(json.dumps({"ts": int(time.time()), "players": players}))
    except Exception:
        pass

@st.cache_data(ttl=3600, show_spinner=False)
def get_active_players() -> List[Dict[str,Any]]:
    # Try local cache first
    cached = _read_players_cache()
    if cached:
        return cached
    # Light single page attempt
    data = fetch_json("players", {"per_page":100, "page":1})
    players = data.get("data",[])
    if players:
        _write_players_cache(players)
    return players

@st.cache_data(ttl=1800, show_spinner=False)
def search_players(query: str) -> List[Dict[str,Any]]:
    q = str(query or "").strip()
    if not q: return []
    data = fetch_json("players", {"per_page":50, "search": q})
    results = data.get("data",[]) or []
    if results:
        # merge into cache
        existing = _read_players_cache()
        d = {p["id"]:p for p in existing}
        for r in results:
            d[r["id"]] = r
        merged = list(d.values())[:400]
        _write_players_cache(merged)
    return results

def build_player_index_from_list(players: List[Dict[str,Any]]):
    labels = []; info={}
    for p in players:
        if not p: continue
        team = p.get("team") or {}
        label = f"{p.get('first_name','')} {p.get('last_name','')} ({team.get('abbreviation','FA')})"
        labels.append(label)
        info[label] = {
            "id": p.get("id"), "first_name": p.get("first_name"), "last_name": p.get("last_name"),
            "team_id": team.get("id"), "team_name": team.get("full_name"), "team_abbr": team.get("abbreviation")
        }
    labels = sorted(list(set(labels)))
    return labels, info

# -----------------------------
# STATS helpers
# -----------------------------
def parse_minutes(min_str:Any) -> float:
    if not min_str: return 0.0
    s = str(min_str)
    if ":" in s:
        try:
            m, sec = s.split(":"); return int(m) + int(sec)/60.0
        except: return 0.0
    try: return float(s)
    except: return 0.0

def prop_value(stat: Dict[str,Any], key:str) -> float:
    pts = stat.get("pts",0) or 0; reb = stat.get("reb",0) or 0; ast = stat.get("ast",0) or 0
    stl = stat.get("stl",0) or 0; blk = stat.get("blk",0) or 0; tov = stat.get("turnover",0) or 0
    fg3m = stat.get("fg3m",0) or 0
    if key=="pts": return pts
    if key=="reb": return reb
    if key=="ast": return ast
    if key=="stl": return stl
    if key=="blk": return blk
    if key=="stl+blk": return stl+blk
    if key=="tov": return tov
    if key=="fg3m": return fg3m
    if key=="pts+reb": return pts+reb
    if key=="pts+ast": return pts+ast
    if key=="reb+ast": return reb+ast
    if key=="pts+reb+ast": return pts+reb+ast
    if key=="min": return parse_minutes(stat.get("min"))
    return 0.0

def average_for_prop(stats: List[Dict[str,Any]], key:str) -> Optional[float]:
    if not stats: return None
    vals = [prop_value(s,key) for s in stats]
    if not vals: return None
    return round(sum(vals)/len(vals),1)

def last_n_average(stats: List[Dict[str,Any]], key:str, n:int) -> Optional[float]:
    if not stats: return None
    stats_sorted = sorted(stats, key=lambda s: s["game"]["date"])
    slice_ = stats_sorted[-n:]
    return average_for_prop(slice_, key)

def get_most_recent_stat(stats: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    if not stats: return None
    return max(stats, key=lambda s: s["game"]["date"])

# API wrappers with caching
@st.cache_data(ttl=900, show_spinner=False)
def get_player_stats_for_season(player_id:int, season:int) -> List[Dict[str,Any]]:
    stats=[]; page=1
    while True:
        data = fetch_json("stats", {"player_ids[]":player_id, "seasons[]":season, "per_page":100, "page":page})
        batch = data.get("data",[]) or []
        stats.extend(batch)
        if len(batch)<100: break
        page+=1
    return stats

@st.cache_data(ttl=900, show_spinner=False)
def get_team_games_for_season(team_id:int, season:int)->List[Dict[str,Any]]:
    games=[]; page=1
    while True:
        data = fetch_json("games", {"team_ids[]":team_id, "seasons[]":season, "per_page":100, "page":page})
        batch = data.get("data",[]) or []
        games.extend(batch)
        if len(batch)<100: break
        page+=1
    return games

@st.cache_data(ttl=900, show_spinner=False)
def get_team_game_on_date(team_id:int, date:dt.date) -> Optional[Dict[str,Any]]:
    date_str = date.strftime("%Y-%m-%d")
    data = fetch_json("games", {"team_ids[]":team_id, "dates[]":date_str, "per_page":100})
    games = data.get("data",[]) or []
    return games[0] if games else None

def get_opponent_from_game(game:Dict[str,Any], team_id:int):
    home = game["home_team"]; visitor = game["visitor_team"]
    if home["id"]==team_id: return visitor, "Home"
    return home, "Away"

@st.cache_data(ttl=300, show_spinner=False)
def get_team_injuries(team_id:int) -> List[Dict[str,Any]]:
    injuries=[]; page=1
    while True:
        data = fetch_json("player_injuries", {"team_ids[]":team_id, "per_page":100, "page":page})
        batch = data.get("data",[]) or []
        injuries.extend(batch)
        if len(batch)<100: break
        page+=1
    return injuries

def injuries_to_df(injuries:List[Dict[str,Any]])->pd.DataFrame:
    rows=[]
    for inj in injuries:
        p = inj.get("player",{})
        rows.append({"Player":f"{p.get('first_name','')} {p.get('last_name','')}", "Status":inj.get("status"), "Return":inj.get("return_date"), "Note":inj.get("description")})
    return pd.DataFrame(rows)

# -----------------------------
# DEFENSIVE RATING CACHE (daily)
# -----------------------------
def load_def_ratings() -> Dict[str, float]:
    # try disk cache
    try:
        if DEF_RATINGS_FILE.exists():
            payload = json.loads(DEF_RATINGS_FILE.read_text())
            ts = payload.get("ts",0)
            if time.time() - ts < 24*3600:
                return payload.get("ratings",{})
    except Exception:
        pass
    # attempt fetch from a reliable free endpoint (we'll use points allowed avg computed from games)
    # To avoid heavy global requests, the getter below will compute points allowed per team only when needed.
    return {}

def save_def_ratings(ratings: Dict[str,float]):
    try:
        DEF_RATINGS_FILE.write_text(json.dumps({"ts":int(time.time()), "ratings":ratings}))
    except Exception:
        pass

@st.cache_data(ttl=3600, show_spinner=False)
def compute_pts_allowed_for_team(team_id:int, season:int) -> Optional[float]:
    games = get_team_games_for_season(team_id, season)
    if not games: return None
    allowed=[]
    for g in games:
        try:
            if g["home_team"]["id"]==team_id:
                opp_score = g.get("visitor_team_score", None)
            else:
                opp_score = g.get("home_team_score", None)
            if opp_score is not None:
                allowed.append(opp_score)
        except Exception:
            continue
    if not allowed: return None
    return float(sum(allowed)/len(allowed))

def map_points_allowed_to_def_score(points_allowed:Optional[float]) -> float:
    if points_allowed is None: return 0.5
    league_avg = 110.0
    diff = points_allowed - league_avg
    offset = (diff/20.0)*0.4
    raw = 0.5 + offset
    return float(max(0.1, min(0.9, raw)))

# -----------------------------
# MINUTES MODEL: training pipeline (builds on first run if MODEL_FILE missing)
# -----------------------------
def build_minutes_dataset(sample_players: List[int], seasons: List[int], max_games=2000) -> pd.DataFrame:
    rows=[]
    for pid in sample_players:
        for season in seasons:
            stats = get_player_stats_for_season(pid, season)
            stats_sorted = sorted(stats, key=lambda s: s["game"]["date"])
            # compute rolling features per game
            mins = [parse_minutes(s.get("min")) for s in stats_sorted]
            if not mins: continue
            for i in range(len(stats_sorted)):
                # target is minutes in this game
                target = mins[i]
                # last1, last5, last10 before this game
                prev = mins[:i]
                last1 = prev[-1] if len(prev)>=1 else None
                last5 = round(sum(prev[-5:])/len(prev[-5:]),1) if prev else None
                last10 = round(sum(prev[-10:])/len(prev[-10:]),1) if prev else None
                season_avg = round(sum(mins)/len(mins),1) if mins else None
                games_played = i
                rows.append({
                    "player_id": pid, "season": season, "games_played": games_played,
                    "last1": last1 or 0.0, "last5": last5 or 0.0, "last10": last10 or 0.0,
                    "season_avg": season_avg or 0.0, "target_min": target
                })
                if len(rows) >= max_games:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)

def train_minutes_model(df: pd.DataFrame):
    # quick clean
    df = df.fillna(0.0)
    X = df[["last1","last5","last10","season_avg","games_played"]].astype(float)
    y = df["target_min"].astype(float)
    if len(df) < 40:
        return None, None
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    return model, mae

def ensure_minutes_model():
    # if model file present, load it
    if MODEL_FILE.exists():
        try:
            model = joblib.load(str(MODEL_FILE))
            return model
        except Exception:
            pass
    # Else train a starter model: pick many players from cached players if possible
    players = get_active_players()
    sample_ids = []
    for p in players[:200]:
        if p and p.get("id"):
            sample_ids.append(p["id"])
    # seasons: last 2 seasons
    today = dt.date.today()
    season = today.year if today.month >= 10 else today.year - 1
    seasons = [season-1, season]
    df = build_minutes_dataset(sample_ids, seasons, max_games=1500)
    if df.empty:
        return None
    model, mae = train_minutes_model(df)
    if model is not None:
        try:
            joblib.dump(model, str(MODEL_FILE))
            joblib.dump({"mae":mae, "trained_ts":int(time.time())}, str(MODEL_META_FILE))
        except Exception:
            pass
    return model

def predict_minutes_with_model(model, stats_recent:List[Dict[str,Any]]):
    # build features: last1,last5,last10,season_avg,games_played
    mins = [parse_minutes(s.get("min")) for s in stats_recent]
    if not mins:
        return None
    last1 = mins[-1] if len(mins)>=1 else 0.0
    last5 = round(sum(mins[-5:])/len(mins[-5:]),1) if len(mins)>=1 else 0.0
    last10 = round(sum(mins[-10:])/len(mins[-10:]),1) if len(mins)>=1 else 0.0
    season_avg = round(sum(mins)/len(mins),1)
    games_played = len(mins)
    X = np.array([[last1, last5, last10, season_avg, games_played]])
    try:
        pred = model.predict(X)[0]
        return float(round(max(0.0, pred),1))
    except Exception:
        return None

# -----------------------------
# PROB/GRADING
# -----------------------------
def normal_cdf(x:float)->float:
    return 0.5*(1+math.erf(x/math.sqrt(2)))

def implied_prob_from_american(odds:Optional[float])->float:
    if odds is None: return 0.535
    if odds<0: return -odds/(-odds+100.0)
    return 100.0/(odds+100.0)

def parse_american_odds(raw:str)->Optional[float]:
    try:
        s = str(raw).strip().replace("+","")
        if s=="": return None
        return float(s)
    except:
        return None

def grade_from_prob_edge(p_hit:Optional[float], edge:Optional[float])->str:
    if p_hit is None or edge is None: return "N/A"
    if p_hit>=0.65 and edge>=0.12: return "A+"
    if p_hit>=0.60 and edge>=0.08: return "A"
    if p_hit>=0.57 and edge>=0.05: return "B+"
    if p_hit>=0.55 and edge>=0.02: return "B"
    if -0.02<=edge<=0.02: return "C"
    if edge < -0.06 and p_hit < 0.52: return "F"
    return "D"

def compute_expected_and_grade(values_recent:List[float], l5:Optional[float], l10:Optional[float], l20:Optional[float],
                              season:Optional[float], line:float, side:str, opp_def_score:float, minutes_adj:float, odds_float:Optional[float]):
    if not values_recent or season is None: return None,None,None,"N/A"
    def safe(v,f): return f if v is None else v
    season_val = season; l10_val = safe(l10, season_val); l5_val = safe(l5, l10_val); l20_val = safe(l20, season_val)
    base_form = 0.45*l10_val + 0.15*l5_val + 0.10*l20_val + 0.30*season_val
    matchup_adj = 1.0 + (opp_def_score-0.5)*0.16
    injury_adj = 1.0 + minutes_adj*0.15
    expected_stat = base_form * matchup_adj * injury_adj
    if len(values_recent)>1:
        std = statistics.pstdev(values_recent)
    else:
        std = max(abs(values_recent[0])/3.0, 0.5)
    std = max(std,0.5)
    if side=="Over":
        z = (line - expected_stat)/std
        p_hit = 1.0 - normal_cdf(z)
    else:
        z = (line - expected_stat)/std
        p_hit = normal_cdf(z)
    p_hit = max(0.01, min(0.99, p_hit))
    book_p = implied_prob_from_american(odds_float)
    edge = p_hit - book_p
    grade = grade_from_prob_edge(p_hit, edge)
    return round(expected_stat,2), round(p_hit,3), round(edge,3), grade

# -----------------------------
# HISTORY persistence
# -----------------------------
def empty_history_df()->pd.DataFrame:
    return pd.DataFrame(columns=[
        "RowID","Date","Player","Player ID","Team","Opponent","Home/Away","Prop","Side","Line","Odds",
        "Season Avg","Last 5 Avg","Last 10 Avg","Last 20 Avg","H2H Avg vs Opp","Game ID",
        "ExpectedStat","HitProb","Edge","Grade","Actual","Result","OppDefScore","PredMinutes"
    ])

def load_history()->pd.DataFrame:
    if not os.path.exists(HISTORY_FILE): return empty_history_df()
    try:
        df = pd.read_csv(HISTORY_FILE)
        for c in empty_history_df().columns:
            if c not in df.columns: df[c]=None
        return df[empty_history_df().columns]
    except:
        return empty_history_df()

def save_history(df:pd.DataFrame):
    df.to_csv(HISTORY_FILE, index=False)

def evaluate_results(df:pd.DataFrame)->pd.DataFrame:
    today = dt.date.today()
    for idx,row in df.iterrows():
        result = str(row.get("Result",""))
        if result in ("Hit","Miss","Push"): continue
        try:
            game_date = dt.datetime.strptime(str(row["Date"]), "%Y-%m-%d").date()
        except:
            continue
        if game_date >= today: continue
        try:
            player_id = int(row["Player ID"]); game_id = int(row["Game ID"])
        except:
            df.at[idx,"Result"]="No game id"
            continue
        prop_key = PROP_DEFS.get(str(row["Prop"]))
        if not prop_key:
            df.at[idx,"Result"]="Unknown prop"
            continue
        stats = fetch_json("stats", {"player_ids[]":player_id, "game_ids[]":game_id, "per_page":10}).get("data",[])
        if not stats:
            df.at[idx,"Result"]="No data"
            continue
        stat = stats[0]
        actual = prop_value(stat, prop_key)
        df.at[idx,"Actual"]=actual
        try:
            line_val = float(row["Line"])
        except:
            df.at[idx,"Result"]="Line error"; continue
        side = str(row.get("Side","Over"))
        if abs(actual-line_val) < 1e-6: res="Push"
        elif side=="Over": res = "Hit" if actual>line_val else "Miss"
        else: res = "Hit" if actual<line_val else "Miss"
        df.at[idx,"Result"]=res
    return df

# -----------------------------
# PARLAY storage
# -----------------------------
def empty_parlay_df()->pd.DataFrame:
    return pd.DataFrame(columns=["ParlayID","Name","Date","LegRowIDs","Stake","BookOdds","ParlayProb","BookImpliedProb","Edge","EV_per_unit"])

def load_parlay_history()->pd.DataFrame:
    if not os.path.exists(PARLAY_FILE): return empty_parlay_df()
    try:
        df = pd.read_csv(PARLAY_FILE)
        for c in empty_parlay_df().columns:
            if c not in df.columns: df[c]=None
        return df[empty_parlay_df().columns]
    except:
        return empty_parlay_df()

def save_parlay_history(df:pd.DataFrame):
    df.to_csv(PARLAY_FILE, index=False)

# -----------------------------
# HEADSHOT helper: try two public endpoints then fallback silhouette
# -----------------------------
SILHOUETTE_DATAURL = None
def make_silhouette_image_bytes(size=(260,260), bgcolor=(10,12,20), fg=(70,70,75)):
    im = Image.new("RGB", size, bgcolor)
    draw = ImageDraw.Draw(im)
    w,h = size
    # draw circle head + shoulders
    cx, cy = w//2, h//3
    r = int(w*0.26)
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fg)
    # shoulders
    draw.rectangle([w*0.18, h*0.55, w*0.82, h*0.88], fill=fg)
    bio = BytesIO(); im.save(bio, format="PNG"); bio.seek(0)
    return bio.read()

SILHOUETTE_BYTES = make_silhouette_image_bytes()

def try_headshot_by_name(first,last):
    # 1) try nba-players.herokuapp.com
    fn = first.lower().replace(" ","_"); ln = last.lower().replace(" ","_")
    urls = [
        f"https://nba-players.herokuapp.com/players/{ln}/{fn}",
        f"https://nba-players.herokuapp.com/players/{ln}/{fn}.png"
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=6)
            if r.status_code==200 and r.content:
                return r.content
        except:
            pass
    # 2) try a secondary generic CDN pattern (may fail for many)
    # (We avoid pulling nba.com directly to prevent auth issues)
    return SILHOUETTE_BYTES

# -----------------------------
# SHARE CARD (dark) generator using PIL
# -----------------------------
def generate_share_card(player_label, prop_label, side, line, odds, season_avg, l5, l10, opp_def_score, pred_min, expected_stat, hitprob, grade):
    W, H = 1000, 420
    bg = (11,15,25); accent=(99,102,241); text_color=(235,238,244); muted=(150,156,170)
    im = Image.new("RGB",(W,H), bg)
    draw = ImageDraw.Draw(im)
    try:
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        font_med = ImageFont.truetype("DejaVuSans.ttf", 20)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font_bold = ImageFont.load_default()
        font_med = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # left: headshot box
    head_box = (26, 26, 26+260, 26+260)

    # try to fetch headshot bytes from player_label
    # player_label typically like "LeBron James (LAL)" so extract first & last
    try:
        parts = player_label.split(" ", 2)
        first = parts[0]
        # last may include team abbrev in parentheses, try safer approach
        last = player_label.replace(first, "").strip().split("(")[0].strip().split(" ")[0]
    except Exception:
        first = player_label
        last = ""

    head_bytes = try_headshot_by_name(first, last)
    try:
        head_img = Image.open(BytesIO(head_bytes)).convert("RGB").resize((260,260))
    except Exception:
        head_img = Image.open(BytesIO(SILHOUETTE_BYTES)).convert("RGB").resize((260,260))

    im.paste(head_img, (26,26))

    # Right: player & prop
    x0 = 310
    draw.text((x0, 36), player_label, font=font_bold, fill=text_color)
    draw.text((x0, 78), f"{prop_label} · {side} {line} ({odds})", font=font_med, fill=muted)

    # metrics: draw vertically with consistent spacing
    metric_y = 130
    metrics = [
        ("Season", season_avg), ("L5", l5), ("L10", l10), ("Opp Def", opp_def_score), ("Pred Min", pred_min)
    ]
    mx = x0
    for i, (lab, val) in enumerate(metrics):
        y = metric_y + i * 32
        val_text = "—" if val is None else str(val)
        draw.text((mx, y), f"{lab}: {val_text}", font=font_small, fill=text_color)

    # bottom section: expected / hit% / grade
    hit_text = "—" if hitprob is None else f"{hitprob*100:.1f}%"
    exp_text = "—" if expected_stat is None else str(expected_stat)
    grade_text = grade or "—"
    draw.text((x0, 300), f"Expected: {exp_text}   Hit%: {hit_text}   Grade: {grade_text}", font=font_med, fill=accent)

    # footer
    draw.text((26, 360), "Built with NBA Prop Research App", font=font_small, fill=muted)

    bio = BytesIO()
    im.save(bio, format="PNG")
    bio.seek(0)
    return bio.getvalue()

# -----------------------------
# UI: session state init
# -----------------------------
if "prop_rows" not in st.session_state:
    st.session_state["prop_rows"] = []

# Build or load minutes model (async in-app training may still block; we do it on demand)
minutes_model = None
try:
    if MODEL_FILE.exists():
        minutes_model = joblib.load(str(MODEL_FILE))
    else:
        # trigger training (may take 10-60s depending on data)
        with st.spinner("Training minutes prediction model (first run; will cache). This can take ~30s..."):
            minutes_model = ensure_minutes_model()
except Exception:
    minutes_model = None

# -----------------------------
# UI: main tabs
# -----------------------------
st.title("NBA Prop Research & Entry")
players_initial = get_active_players()
if players_initial:
    player_labels, player_info = build_player_index_from_list(players_initial)
else:
    player_labels, player_info = [], {}

tab_form, tab_research, tab_history, tab_parlay = st.tabs(["Prop Entry Form","Player Research","Prop History","Parlay Builder"])

# -----------------------------
# TAB 1: Prop Entry
# -----------------------------
with tab_form:
    st.subheader("Daily Prop Entry")
    with st.form(key="prop_entry_form", clear_on_submit=False):
        game_date = st.date_input("Game date", value=dt.date.today())
        player_search = st.text_input("Search player (type 2+ letters)", placeholder="LeBron")
        player_options = [""]
        if player_search and len(player_search.strip())>=2:
            results = search_players(player_search.strip())
            labels, info_map = build_player_index_from_list(results)
            if labels:
                player_options = [""] + labels
                player_info.update(info_map)
                for lab in labels:
                    if lab not in player_labels: player_labels.append(lab)
        player_label = st.selectbox("Player", options=player_options, index=0)
        if player_label and player_label in player_info:
            info = player_info[player_label]
            st.markdown(f"<div style='display:flex;align-items:center;gap:12px'>", unsafe_allow_html=True)
            # headshot
            try:
                first = info["first_name"]; last = info["last_name"]
                head_bytes = try_headshot_by_name(first,last)
    if head_bytes:
    b64 = base64.b64encode(head_bytes).decode("utf-8")
    st.markdown(f"<div class='player-photo-card'><img src='data:image/png;base64,{b64}'/></div>", unsafe_allow_html=True)
    else:
    # fallback silhouette
    b64 = base64.b64encode(SILHOUETTE_BYTES).decode("utf-8")
    st.markdown(f"<div class='player-photo-card'><img src='data:image/png;base64,{b64}'/></div>", unsafe_allow_html=True)



            except Exception:
                st.markdown(f"<div class='player-photo-card'><img src='data:image/png;base64,'/></div>", unsafe_allow_html=True)
            st.markdown(f"<div><b>{info['first_name']} {info['last_name']}</b><br/>{info['team_name']} ({info['team_abbr']})</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        prop_choice = st.selectbox("Prop", options=list(PROP_DEFS.keys()), index=0)
        side_choice = st.selectbox("Side", options=["Over","Under"], index=0)
        line_value = st.number_input("Line (number)", step=0.5, format="%.2f")
        odds_str = st.text_input("Odds (optional, e.g. -115)", value="")
        st.markdown("Opponent defense and predicted minutes are computed automatically.")
        col1, col2 = st.columns([1,1])
        with col1:
            add_pressed = st.form_submit_button("➕ Add to prop sheet")
        with col2:
            clear_pressed = st.form_submit_button("🧹 Clear inputs")
    if 'add_pressed' in locals() and add_pressed:
        if not (player_label and player_label in player_info):
            st.warning("Pick a player first.")
        else:
            info = player_info[player_label]
            pid = info["id"]; team_id = info["team_id"]; team_name = info["team_name"]
            game = get_team_game_on_date(team_id, game_date)
            opp_name="Unknown"; home_away="N/A"; opp_id=None; game_id=None
            if game:
                opp_team, home_away = get_opponent_from_game(game, team_id)
                opp_name = opp_team["full_name"]; opp_id = opp_team["id"]; game_id = game["id"]
            today = dt.date.today(); season = get_current_season(today); prev_season = season-1
            stats_current = get_player_stats_for_season(pid, season)
            stats_prev = get_player_stats_for_season(pid, prev_season)
            stats_recent = sorted(stats_prev + stats_current, key=lambda s: s["game"]["date"])
            prop_key = PROP_DEFS[prop_choice]
            season_avg = average_for_prop(stats_current, prop_key)
            last5_avg = last_n_average(stats_recent, prop_key, 5)
            last10_avg = last_n_average(stats_recent, prop_key, 10)
            last20_avg = last_n_average(stats_recent, prop_key, 20)
            # Opp def score
            opp_def_score = 0.5
            if opp_id:
                pts_allowed = compute_pts_allowed_for_team(opp_id, season)
                opp_def_score = map_points_allowed_to_def_score(pts_allowed)
            # predicted minutes: prefer ML model; fallback to heuristic
            pred_min = None; minutes_adj = 0.0
            try:
                if minutes_model:
                    pm = predict_minutes_with_model(minutes_model, stats_recent)
                    if pm is not None:
                        pred_min = pm
                if pred_min is None:
                    # heuristic: last10 min bumped by injuries
                    last10min = last_n_average(stats_recent, "min", 10) or average_for_prop(stats_current, "min") or 0.0
                    inj = 0
                    try:
                        injlist = get_team_injuries(team_id)
                        for x in injlist:
                            stext=(x.get("status") or "").lower()
                            if stext and ("out" in stext or "questionable" in stext or "day_to_day" in stext or "doubtful" in stext):
                                inj += 1
                    except:
                        inj = 0
                    bump = min(0.25, 0.08 * inj)
                    pred_min = round(last10min * (1.0 + bump),1)
                # minutes_adj normalization: relative to season avg
                season_min_avg = average_for_prop(stats_current, "min") or pred_min or 0.0
                denom = 12.0
                minutes_adj = (pred_min - (season_min_avg or pred_min)) / denom if denom else 0.0
                minutes_adj = max(-1.0, min(1.0, minutes_adj))
            except Exception:
                pred_min = None; minutes_adj = 0.0
            # compute expected/hitprob/edge/grade
            values_recent = [prop_value(s,prop_key) for s in (stats_recent[-20:] if len(stats_recent)>20 else stats_recent)]
            odds_float = parse_american_odds(odds_str)
            if values_recent and season_avg is not None:
                expected_stat, hit_prob, edge, grade = compute_expected_and_grade(values_recent,last5_avg,last10_avg,last20_avg,
                                                                                  season_avg,float(line_value),side_choice,opp_def_score,minutes_adj,odds_float)
            else:
                expected_stat = hit_prob = edge = None; grade="N/A"
            row_id = str(uuid.uuid4())
            row = {
                "RowID": row_id, "Date": game_date.strftime("%Y-%m-%d"), "Player": player_label, "Player ID": pid,
                "Team": team_name, "Opponent": opp_name, "Home/Away": home_away, "Prop": prop_choice, "Side": side_choice,
                "Line": float(line_value), "Odds": odds_str, "Season Avg": season_avg, "Last 5 Avg": last5_avg,
                "Last 10 Avg": last10_avg, "Last 20 Avg": last20_avg, "H2H Avg vs Opp": None, "Game ID": game_id,
                "ExpectedStat": expected_stat, "HitProb": hit_prob, "Edge": edge, "Grade": grade, "Actual": None, "Result": "Pending",
                "OppDefScore": round(opp_def_score,2) if opp_def_score is not None else None, "PredMinutes": pred_min
            }
            st.session_state["prop_rows"].append(row)
            hist_df = load_history()
            hist_df = pd.concat([hist_df, pd.DataFrame([row])], ignore_index=True)
            save_history(hist_df)
            st.success("Prop added to sheet and history.")

    if 'clear_pressed' in locals() and clear_pressed:
        st.experimental_rerun()

    st.markdown("### Current prop sheet (compact)")
    if st.session_state["prop_rows"]:
        df_props = pd.DataFrame(st.session_state["prop_rows"])
        if "RowID" not in df_props.columns:
            df_props["RowID"] = [str(uuid.uuid4()) for _ in range(len(df_props))]
            for i,rid in df_props["RowID"].items():
                st.session_state["prop_rows"][i]["RowID"] = rid
        else:
            for i in df_props.index:
                if pd.isna(df_props.at[i,"RowID"]) or df_props.at[i,"RowID"]=="":
                    new_id=str(uuid.uuid4()); df_props.at[i,"RowID"]=new_id; st.session_state["prop_rows"][i]["RowID"]=new_id
        df_display = df_props.copy(); df_display.index.name = "Row"
        cols = ["Player","Prop","Side","Line","Odds","PredMinutes","OppDefScore","ExpectedStat","HitProb","Edge","Grade","Result"]
        cols = [c for c in cols if c in df_display.columns]
        st.dataframe(df_display[cols], use_container_width=True, height=320)
        del_col1, del_col2 = st.columns([2,1])
        with del_col1:
            del_idx = st.selectbox("Select row to delete", options=df_display.index,
                                   format_func=lambda i: f"{i}: {df_display.loc[i,'Player']} · {df_display.loc[i,'Prop']} {df_display.loc[i,'Line']}")
        with del_col2:
            if st.button("Delete selected row"):
                row_to_delete = df_props.loc[del_idx]
                st.session_state["prop_rows"].pop(int(del_idx))
                hist_df = load_history()
                rid = row_to_delete.get("RowID")
                if rid and "RowID" in hist_df.columns:
                    hist_df = hist_df[hist_df["RowID"] != rid]; save_history(hist_df)
                st.experimental_rerun()
        st.markdown("### Grouped by game")
        df_group = df_props.copy()
        df_group["GameLabel"] = df_group.apply(lambda r: f"{r['Date']} · {r['Team']} vs {r['Opponent']} ({r['Home/Away']})", axis=1)
        for game_label, gdf in df_group.groupby("GameLabel"):
            with st.expander(game_label, expanded=False):
                show_cols = ["Player","Prop","Side","Line","Odds","PredMinutes","OppDefScore","ExpectedStat","HitProb","Grade"]
                show_cols = [c for c in show_cols if c in gdf.columns]
                st.dataframe(gdf[show_cols], use_container_width=True)
    else:
        st.info("No props added yet.")

# -----------------------------
# TAB 2: Player Research
# -----------------------------
with tab_research:
    st.subheader("Player Research Lab")
    rcol1, rcol2 = st.columns([1.2,1])
    with rcol1:
        research_search = st.text_input("Search player (2+ letters)")
    with rcol2:
        research_prop_choice = st.selectbox("Prop focus", options=list(PROP_DEFS.keys()), index=0)
    research_player_label = ""
    if research_search and len(research_search.strip())>=2:
        results = search_players(research_search.strip())
        labels, info_map = build_player_index_from_list(results)
        for lab in labels:
            if lab not in player_labels: player_labels.append(lab)
        player_info.update(info_map)
        if labels:
            research_player_label = st.selectbox("Select player", options=[""]+labels, index=0)
        else:
            st.info("No players found.")
    else:
        research_player_label = st.selectbox("Player to research", options=[""]+player_labels, index=0)
    if research_player_label and research_player_label in player_info:
        info = player_info[research_player_label]; pid = info["id"]; team_id = info["team_id"]
        st.image(BytesIO(try_headshot_by_name(info["first_name"], info["last_name"])), width=130)
        st.markdown(f"**{info['first_name']} {info['last_name']}**  \n{info['team_name']} ({info['team_abbr']}) · ID `{pid}`")
        today = dt.date.today(); season = get_current_season(today); prev_season=season-1
        prop_key = PROP_DEFS[research_prop_choice]
        stats_current = get_player_stats_for_season(pid, season); stats_prev = get_player_stats_for_season(pid, prev_season)
        stats_recent = sorted(stats_prev + stats_current, key=lambda s: s["game"]["date"])
        if not stats_recent:
            st.warning("No stats found.")
        else:
            most_recent = get_most_recent_stat(stats_recent)
            last10_avg = last_n_average(stats_recent, prop_key, 10)
            last20_avg = last_n_average(stats_recent, prop_key, 20)
            season_avg = average_for_prop(stats_current, prop_key)
            prev_season_avg = average_for_prop(stats_prev, prop_key)
            next_game = get_team_game_on_date(team_id, dt.date.today())
            opp_team = None; opp_id=None; h2h_avg=None; h2h_games=0
            if next_game:
                opp_team, home_away = get_opponent_from_game(next_game, team_id); opp_id=opp_team["id"]
                st.markdown(f"**Next Game:** {next_game['date']}  ·  {info['team_abbr']} vs {opp_team['abbreviation']} ({home_away})")
                stats_for_h2h = get_player_stats_for_season(pid, season) + get_player_stats_for_season(pid, prev_season)
                stats_h2h = [s for s in stats_for_h2h if s["game"]["home_team_id"]==opp_id or s["game"]["visitor_team_id"]==opp_id]
                h2h_games = len(stats_h2h); h2h_avg = average_for_prop(stats_h2h, prop_key)
            st.markdown("##### Form snapshot for this prop")
            f1,f2,f3,f4,f5 = st.columns(5)
            def metric_card(col,label,value,extra=""):
                display = "—" if value is None else value
                col.markdown(f"<div class='metric-card'><div class='metric-tag'>{label}</div><div class='metric-value'>{display}</div><div style='font-size:0.7rem;color:#9ca3af;margin-top:2px'>{extra}</div></div>", unsafe_allow_html=True)
            metric_card(f1,"Most recent", prop_value(most_recent, prop_key), most_recent["game"]["date"])
            metric_card(f2,"Last 10 avg", last10_avg); metric_card(f3,"Last 20 avg", last20_avg)
            metric_card(f4,f"{season} season avg", season_avg); metric_card(f5,f"{prev_season} season avg", prev_season_avg)
            if opp_team:
                st.markdown("##### H2H vs next opponent")
                hcol1,hcol2 = st.columns([1,3])
                metric_card(hcol1,"H2H avg", h2h_avg, extra=f"{h2h_games} games (last 2 seasons)")
                if h2h_games:
                    rows=[]
                    for s in sorted(stats_h2h, key=lambda s: s["game"]["date"], reverse=True):
                        g=s["game"]; rows.append({"Date":g["date"], "Prop Value":prop_value(s,prop_key), "PTS":s.get("pts",0),"REB":s.get("reb",0),"AST":s.get("ast",0),"3PM":s.get("fg3m",0),"MIN":parse_minutes(s.get("min"))})
                    hcol2.dataframe(pd.DataFrame(rows), use_container_width=True, height=240)
            st.markdown("##### Injury report (may impact minutes)")
            icol1,icol2 = st.columns(2)
            team_inj = get_team_injuries(team_id)
            if team_inj: icol1.dataframe(injuries_to_df(team_inj), use_container_width=True, height=220)
            else: icol1.info("No current injuries listed.")
            if opp_team:
                opp_inj = get_team_injuries(opp_id)
                if opp_inj: icol2.dataframe(injuries_to_df(opp_inj), use_container_width=True, height=220)
                else: icol2.info("No current injuries listed for opponent.")

# -----------------------------
# TAB 3: Prop History
# -----------------------------
with tab_history:
    st.subheader("Prop History & Results")
    hist_df = load_history()
    # recompute missing metrics for recent rows
    hist_df = hist_df.copy()
    # run evaluation/backfill on demand
    if hist_df.empty:
        st.info("No history saved yet. Add props on the form tab.")
    else:
        hist_df["Date_dt"] = pd.to_datetime(hist_df["Date"], errors="coerce")
        min_d = hist_df["Date_dt"].min().date(); max_d = hist_df["Date_dt"].max().date()
        date_range = st.date_input("Filter by date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        if isinstance(date_range, tuple):
            start_d, end_d = date_range
        else:
            start_d = end_d = date_range
        mask = (hist_df["Date_dt"].dt.date >= start_d) & (hist_df["Date_dt"].dt.date <= end_d)
        view_df = hist_df.loc[mask].drop(columns=["Date_dt"]).sort_values("Date", ascending=False)
        show_cols = ["Date","Player","Team","Opponent","Prop","Side","Line","Odds","Season Avg","Last 5 Avg","Last 10 Avg","Last 20 Avg","OppDefScore","PredMinutes","ExpectedStat","HitProb","Edge","Grade","Actual","Result"]
        show_cols = [c for c in show_cols if c in view_df.columns]
        st.dataframe(view_df[show_cols], use_container_width=True, height=420)
        c1,c2 = st.columns([1,2])
        with c1:
            if st.button("Update results for completed games"):
                updated = evaluate_results(hist_df)
                save_history(updated)
                st.success("Results updated from BallDontLie stats."); st.experimental_rerun()
        csv_hist = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download full history CSV", data=csv_hist, file_name="prop_history.csv", mime="text/csv")

# -----------------------------
# TAB 4: Parlay Builder
# -----------------------------
with tab_parlay:
    st.subheader("Parlay Builder")
    if not st.session_state["prop_rows"]:
        st.info("No props in the current sheet.")
    else:
        df_props = pd.DataFrame(st.session_state["prop_rows"])
        if "RowID" not in df_props.columns:
            df_props["RowID"] = [str(uuid.uuid4()) for _ in range(len(df_props))]
            for i,rid in df_props["RowID"].items():
                st.session_state["prop_rows"][i]["RowID"] = rid
        df_props.index.name="Row"
        leg_indices = st.multiselect("Select legs", options=df_props.index.tolist(),
                                     format_func=lambda i: f"{i}: {df_props.loc[i,'Player']} · {df_props.loc[i,'Prop']} {df_props.loc[i,'Side']} {df_props.loc[i,'Line']} ({df_props.loc[i,'Odds']})")
        pcol1,pcol2,pcol3 = st.columns([1,1,1])
        with pcol1: parlay_name = st.text_input("Parlay name","")
        with pcol2: stake = st.number_input("Stake (units)", min_value=0.0, value=1.0, step=0.5)
        with pcol3: parlay_odds_str = st.text_input("Book parlay odds (e.g. +600)","")
        parlay_prob = ev_per_unit = parlay_edge = None
        if leg_indices:
            leg_probs=[]
            for i in leg_indices:
                p = df_props.loc[i].get("HitProb")
                if p is None or pd.isna(p): p = 0.5
                leg_probs.append(float(p))
            parlay_prob = 1.0
            for p in leg_probs: parlay_prob *= p
            parlay_prob = max(0.0001, min(0.999, parlay_prob))
            parlay_odds = parse_american_odds(parlay_odds_str)
            book_p = implied_prob_from_american(parlay_odds)
            if parlay_odds is not None:
                dec = (1 + (parlay_odds/100.0)) if parlay_odds>0 else (1 + (100.0/(-parlay_odds)))
                ev_per_unit = parlay_prob*(dec-1.0) - (1.0-parlay_prob)
            parlay_edge = parlay_prob - book_p
            st.markdown("##### Parlay summary")
            pm1,pm2,pm3 = st.columns(3)
            def metric_card_small(col,label,val): col.markdown(f"<div class='metric-card'><div class='metric-tag'>{label}</div><div class='metric-value'>{val}</div></div>", unsafe_allow_html=True)
            metric_card_small(pm1,"Model parlay prob", f"{parlay_prob*100:.2f}%" if parlay_prob else "—")
            metric_card_small(pm2,"Book implied prob", f"{book_p*100:.2f}%" if parlay_odds is not None else "—")
            metric_card_small(pm3,"Parlay EV", round(ev_per_unit,3) if ev_per_unit is not None else "—")
            if st.button("Save parlay"):
                parlay_id = str(uuid.uuid4()); leg_row_ids = ";".join([str(df_props.loc[i].get("RowID")) for i in leg_indices])
                row={"ParlayID":parlay_id,"Name":parlay_name,"Date":dt.date.today().strftime("%Y-%m-%d"),"LegRowIDs":leg_row_ids,"Stake":stake,"BookOdds":parlay_odds_str,"ParlayProb":parlay_prob,"BookImpliedProb":book_p,"Edge":parlay_edge,"EV_per_unit":ev_per_unit}
                p_hist = load_parlay_history(); p_hist = pd.concat([p_hist,pd.DataFrame([row])], ignore_index=True); save_parlay_history(p_hist); st.success("Parlay saved.")
        st.markdown("### Parlay history")
        p_hist = load_parlay_history()
        if p_hist.empty: st.info("No parlays saved yet.")
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
                    if pd.isna(leg_ids_str) or str(leg_ids_str).strip()=="":
                        st.write("No leg IDs stored."); continue
                    leg_ids=[x for x in str(leg_ids_str).split(";") if x and x.lower()!="nan"]
                    if not leg_ids: st.write("No valid leg IDs stored."); continue
                    df_legs = hist_df[hist_df["RowID"].astype(str).isin(leg_ids)]
                    if df_legs.empty: st.write("No matching props found in history for these legs.")
                    else:
                        st.dataframe(df_legs[["Date","Player","Team","Opponent","Prop","Side","Line","Odds","ExpectedStat","HitProb","Grade","Result"]], use_container_width=True, height=260)
