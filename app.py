# BMTC Bus Delay Predictor — Bengaluru
# Option C: XGBoost (all stops) + Prophet (top 30 high-delay stops)
# NEW: Live OpenWeatherMap rain detection + current IST time as default

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os
import re
import requests
from difflib import get_close_matches
from datetime import date, datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

if "user" not in st.session_state:
    st.session_state.user = None

if "history" not in st.session_state:
    st.session_state.history = []

if "favorites" not in st.session_state:
    st.session_state.favorites = []

# ---------------- USER AUTH SYSTEM ----------------
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        return json.load(open(USERS_FILE))
    return {}

def save_users(users):
    json.dump(users, open(USERS_FILE, "w"))

def login(username, password):
    users = load_users()
    if username in users and users[username] == password:
        st.session_state.user = username
        return True
    return False

def signup(username, password):
    users = load_users()
    users[username] = password
    save_users(users)

# ---------------- LOGIN UI ----------------
if not st.session_state.user:
    st.title("🔐 Login to BMTC Smart App")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if login(u, p):
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_u = st.text_input("New Username")
        new_p = st.text_input("New Password", type="password")

        if st.button("Signup"):
            signup(new_u, new_p)
            st.success("Account created! Please login.")

    st.stop()

# ---------------- HEADER ----------------
st.title(f"🚌 Welcome {st.session_state.user}")

# ---------------- EXISTING MODEL FUNCTIONS ----------------
# (Keep ALL your original ML + GTFS code here unchanged)

def predict_delay_stub():
    return np.random.randint(1, 15)

# ---------------- ETA CALCULATION ----------------
def calculate_eta(delay, base_time=30):
    return base_time + delay

# ---------------- MAIN TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Predict",
    "⭐ Favorites",
    "📜 History",
    "👤 Profile"
])

# ==========================================================
# TAB 1 — PREDICTION (ENHANCED)
# ==========================================================
with tab1:
    st.subheader("Plan Your Journey")

    src = st.text_input("From")
    dst = st.text_input("To")

    if st.button("Predict Journey"):
        delay = predict_delay_stub()
        eta = calculate_eta(delay)

        st.success(f"Predicted Delay: {delay} min")
        st.info(f"Estimated Travel Time (ETA): {eta} min")

        # SAVE HISTORY
        st.session_state.history.append({
            "from": src,
            "to": dst,
            "delay": delay,
            "eta": eta,
            "time": str(datetime.now())
        })

        # ADD TO FAVORITES
        if st.button("⭐ Add to Favorites"):
            st.session_state.favorites.append({
                "from": src,
                "to": dst
            })
            st.success("Added to favorites!")

with tab2:
    st.subheader("⭐ Your Favorite Routes")

    if not st.session_state.favorites:
        st.info("No favorites yet")
    else:
        for fav in st.session_state.favorites:
            col1, col2 = st.columns([3,1])

            with col1:
                st.write(f"📍 {fav['from']} → {fav['to']}")

            with col2:
                if st.button("Use", key=f"{fav['from']}_{fav['to']}"):
                    st.session_state.selected_fav = fav
                    st.success("Loaded into predictor!")


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMTC Delay Predictor",
    page_icon="🚌",
    layout="centered"
)

# ── File paths ────────────────────────────────────────────────────────────────
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"

# ── OpenWeatherMap API key ────────────────────────────────────────────────────
# Get a free key at https://openweathermap.org/api (Free tier: 60 calls/min)
# Add it to Streamlit Cloud: Settings → Secrets → paste exactly as shown below:
#
#   [openweather]
#   api_key = "your_actual_key_here"
#
# NOTE: st.secrets does NOT support chained .get() on nested keys.
# Always use try/except for safe nested secret access.
try:
    OWM_API_KEY = st.secrets["openweather"]["api_key"]
except (KeyError, FileNotFoundError):
    OWM_API_KEY = ""

# ── Live weather fetch ────────────────────────────────────────────────────────
@st.cache_data(ttl=600)   # refresh every 10 minutes
def fetch_bengaluru_weather():
    """
    Returns (is_rain, description, temp_c, icon_url).
    Falls back to (None, error_msg, None, None) if API key missing or call fails.
    OWM weather IDs: < 700 = rain/drizzle/thunderstorm/snow
    """
    if not OWM_API_KEY:
        return None, "no_key", None, None
    try:
        url    = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat"   : 12.9716,    # Bengaluru city centre
            "lon"   : 77.5946,
            "appid" : OWM_API_KEY,
            "units" : "metric",
        }
        r    = requests.get(url, params=params, timeout=5)
        data = r.json()
        wid  = data["weather"][0]["id"]
        desc = data["weather"][0]["description"].capitalize()
        temp = round(data["main"]["temp"], 1)
        icon = f"https://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png"
        return int(wid < 700), desc, temp, icon
    except Exception as e:
        return None, str(e), None, None

# ── Current IST time helper ───────────────────────────────────────────────────
def get_ist_now():
    """Returns current datetime in Asia/Kolkata timezone."""
    return datetime.now(pytz.timezone("Asia/Kolkata"))

# ── Build list of 30-min time slots ──────────────────────────────────────────
def build_time_slots():
    """
    Returns list of labels like ['00:00','00:30','01:00', ... '23:30']
    and a matching list of (hour, minute) tuples.
    """
    labels = []
    values = []
    for h in range(24):
        for m in (0, 30):
            labels.append(f"{h:02d}:{m:02d}")
            values.append((h, m))
    return labels, values

TIME_LABELS, TIME_VALUES = build_time_slots()   # 48 slots

def slot_index_for(hour, minute):
    """Find the nearest 30-min slot index for a given hour:minute."""
    target = hour * 60 + minute
    # round to nearest 30
    rounded = round(target / 30) * 30
    rounded = min(rounded, 23 * 60 + 30)   # cap at 23:30
    h, m = divmod(rounded, 60)
    try:
        return TIME_VALUES.index((h, m))
    except ValueError:
        return 0

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    xgb_model    = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    stop_summary = pd.read_csv(os.path.join(MODEL_DIR, "stop_summary.csv"))
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    return xgb_model, stop_summary, metadata

@st.cache_data
def load_results():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "final_results.csv"))

@st.cache_data
def load_prophet_stops():
    path = os.path.join(MODEL_DIR, "prophet_stops.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

@st.cache_resource
def load_prophet_model(stop_name):
    from prophet.serialize import model_from_json
    safe = stop_name.replace(' ','_').replace('/','_').replace('(','').replace(')','')
    path = os.path.join(MODEL_DIR, "prophet", f"{safe}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return model_from_json(f.read())

# routes.csv / load_routes() removed — bus lookup now uses GTFS join (models/gtfs/)

# ── Load everything at startup ────────────────────────────────────────────────
try:
    xgb_model, stop_summary, metadata = load_assets()
    final_results = load_results()
    prophet_stops = load_prophet_stops()
    FEATURES      = metadata["features"]
    all_stops     = sorted(stop_summary["stop_name"].tolist())
except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.info(
        "Make sure you have run all three notebooks and copied the "
        "`models/` and `outputs/` folders into the root of your GitHub repo."
    )
    st.stop()

# ── Helpers ───────────────────────────────────────────────────────────────────
def find_stop(query, n=6):
    query = query.lower().strip()
    exact = [s for s in all_stops if query in s.lower()]
    if exact:
        return exact[:n]
    fuzzy = get_close_matches(query, [s.lower() for s in all_stops],
                              n=n, cutoff=0.4)
    return [s for s in all_stops if s.lower() in fuzzy]

# TAB 3 — HISTORY
# ==========================================================
with tab3:
    st.subheader("📜 Travel History")

    if not st.session_state.history:
        st.info("No history yet")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

# ══════════════════════════════════════════════════════════════════════════════
# BUS NUMBER LOOKUP  —  GTFS proper join pipeline
# ══════════════════════════════════════════════════════════════════════════════
#
#  HOW IT WORKS (stop_name → bus numbers):
#
#   stops.txt       stop_name  ──►  stop_id
#       ↓
#   stop_times.txt  stop_id    ──►  trip_id  (all trips that call at this stop)
#       ↓
#   trips.txt       trip_id    ──►  route_id
#       ↓
#   routes.txt      route_id   ──►  route_short_name  (the bus number e.g. "335E")
#
#  For a FROM→TO pair we do this for both stops, then intersect the two sets
#  of route_short_names → those are the buses that directly serve both stops.
#
#  FILES REQUIRED (place in models/gtfs/):
#    stops.txt, stop_times.txt, trips.txt, routes.txt
#
#  If the gtfs/ folder is absent the app falls back gracefully with a message.
# ──────────────────────────────────────────────────────────────────────────────

GTFS_DIR = os.path.join(MODEL_DIR, "gtfs")

@st.cache_resource(show_spinner=False)
def load_gtfs_tables():
    """
    Load the four GTFS text files and build a dict:
      stop_name_lower  →  set of route_short_name strings

    Returns (lookup_dict, True) on success, ({}, False) if files are missing.
    The heavy join is done once at startup and cached for the whole session.
    """
    required = ["stops.txt", "stop_times.txt", "trips.txt", "routes.txt"]
    paths    = {f: os.path.join(GTFS_DIR, f) for f in required}

    if not all(os.path.exists(p) for p in paths.values()):
        return {}, False

    # ── 1. stops: stop_id → stop_name ────────────────────────────────────────
    stops = pd.read_csv(
        paths["stops.txt"],
        usecols=["stop_id", "stop_name"],
        dtype=str,
    ).dropna()
    stops["stop_name_lower"] = stops["stop_name"].str.lower().str.strip()
    stop_id_to_name = stops.set_index("stop_id")["stop_name"].to_dict()

    # ── 2. routes: route_id → route_short_name ───────────────────────────────
    routes = pd.read_csv(
        paths["routes.txt"],
        usecols=["route_id", "route_short_name"],
        dtype=str,
    ).dropna()
    route_id_to_short = routes.set_index("route_id")["route_short_name"].to_dict()

    # ── 3. trips: trip_id → route_id ─────────────────────────────────────────
    trips = pd.read_csv(
        paths["trips.txt"],
        usecols=["trip_id", "route_id"],
        dtype=str,
    ).dropna()
    trip_id_to_route = trips.set_index("trip_id")["route_id"].to_dict()

    # ── 4. stop_times: stop_id → set of trip_ids ─────────────────────────────
    # stop_times.txt is the largest file — read only the two columns we need
    stop_times = pd.read_csv(
        paths["stop_times.txt"],
        usecols=["trip_id", "stop_id"],
        dtype=str,
    ).dropna()

    # ── 5. Build final lookup: stop_name_lower → set[route_short_name] ───────
    # Merge stop_times → trips → routes in one vectorised pass
    stop_times["route_id"] = stop_times["trip_id"].map(trip_id_to_route)
    stop_times["route_short_name"] = stop_times["route_id"].map(route_id_to_short)
    stop_times["stop_name_lower"] = stop_times["stop_id"].map(
        stops.set_index("stop_id")["stop_name_lower"]
    )
    stop_times = stop_times.dropna(
        subset=["stop_name_lower", "route_short_name"]
    )

    lookup: dict[str, set] = {}
    for row in stop_times[["stop_name_lower", "route_short_name"]].itertuples(index=False):
        lookup.setdefault(row.stop_name_lower, set()).add(row.route_short_name)

    return lookup, True


def _best_gtfs_match(query: str, lookup: dict) -> str | None:
    """
    Find the best matching stop_name_lower key in the GTFS lookup dict.
    Tries exact substring first, then difflib fuzzy match.
    Returns the matched key or None.
    """
    q = query.lower().strip()
    # Exact match
    if q in lookup:
        return q
    # Substring match (query is contained in a gtfs stop name)
    substr = [k for k in lookup if q in k]
    if substr:
        # prefer the shortest match (most specific)
        return min(substr, key=len)
    # Reverse substring (gtfs stop name is contained in query)
    substr2 = [k for k in lookup if k in q]
    if substr2:
        return max(substr2, key=len)
    # Fuzzy fallback
    matches = get_close_matches(q, list(lookup.keys()), n=1, cutoff=0.6)
    return matches[0] if matches else None


def find_buses(src: str, dst: str) -> dict:
    """
    Look up bus numbers that serve both src and dst stops using GTFS data.

    Returns:
      {
        "buses"  : list[str],   # route_short_names serving both stops
        "source" : "gtfs" | "none",
        "note"   : str,
        "hops"   : int,         # 1 = direct routes found
      }
    """
    lookup, gtfs_ok = load_gtfs_tables()

    if not gtfs_ok:
        # GTFS files not present — inform the user clearly
        return {
            "buses" : [],
            "source": "none",
            "note"  : (
                "GTFS files not found in models/gtfs/. "
                "Add stops.txt, stop_times.txt, trips.txt, routes.txt "
                "to enable bus number lookup."
            ),
            "hops"  : 0,
        }

    src_key = _best_gtfs_match(src, lookup)
    dst_key = _best_gtfs_match(dst, lookup)

    if not src_key:
        return {
            "buses" : [],
            "source": "none",
            "note"  : f"Stop not found in GTFS: '{src}'. Try a nearby landmark name.",
            "hops"  : 0,
        }
    if not dst_key:
        return {
            "buses" : [],
            "source": "none",
            "note"  : f"Stop not found in GTFS: '{dst}'. Try a nearby landmark name.",
            "hops"  : 0,
        }

    src_routes = lookup[src_key]   # all routes serving FROM stop
    dst_routes = lookup[dst_key]   # all routes serving TO stop

    # Direct buses = routes that call at BOTH stops
    direct = sorted(src_routes & dst_routes)

    if direct:
        return {
            "buses" : direct[:10],   # cap at 10 to keep UI clean
            "source": "gtfs",
            "note"  : (
                f"BMTC GTFS data · matched '{src_key}' → '{dst_key}' · "
                f"{len(direct)} direct route(s) found"
            ),
            "hops"  : 1,
        }

    # No direct buses — show routes serving each stop separately as a hint
    src_list = sorted(src_routes)[:5]
    dst_list = sorted(dst_routes)[:5]
    return {
        "buses" : [],
        "source": "none",
        "note"  : (
            f"No direct bus found. "
            f"Buses at {src}: {', '.join(src_list) or 'none'}. "
            f"Buses at {dst}: {', '.join(dst_list) or 'none'}. "
            "You may need to transfer."
        ),
        "hops"  : 2,
    }

def build_features(stop_name, hour, dow, month, is_rain):
    row = stop_summary[stop_summary["stop_name"] == stop_name]
    if row.empty:
        return None, None
    s     = row.iloc[0]
    avg_d = float(s["avg_delay"])
    is_weekend = int(dow >= 5)
    is_rush    = int(hour in [7, 8, 9, 17, 18, 19])
    hour_sin   = np.sin(2 * np.pi * hour / 24)
    hour_cos   = np.cos(2 * np.pi * hour / 24)
    dow_sin    = np.sin(2 * np.pi * dow  / 7)
    dow_cos    = np.cos(2 * np.pi * dow  / 7)
    input_dict = {
        "factor"      : float(s["factor"]),
        "trip_count"  : float(s["trip_count"]),
        "route_count" : float(s["route_count"]),
        "hour"        : hour,
        "day_of_week" : dow,
        "month"       : month,
        "is_weekend"  : is_weekend,
        "is_rush"     : is_rush,
        "is_rain"     : is_rain,
        "hour_sin"    : hour_sin,
        "hour_cos"    : hour_cos,
        "dow_sin"     : dow_sin,
        "dow_cos"     : dow_cos,
        "lag_1h"      : avg_d,
        "lag_24h"     : avg_d,
        "roll_3h"     : avg_d,
        "roll_6h"     : avg_d,
        "roll_24h"    : avg_d,
    }
    X = pd.DataFrame([input_dict])[FEATURES]
    return X, s

def predict_delay(stop_name, hour, dow, month, is_rain):
    X, _ = build_features(stop_name, hour, dow, month, is_rain)
    if X is None:
        return 0.0
    return round(float(np.clip(xgb_model.predict(X)[0], 0, None)), 1)

def get_status(delay):
    if delay < 3:  return "✅ On Time",      "#065F46", "#D1FAE5"
    if delay < 8:  return "⚠️ Minor Delay",  "#92400E", "#FEF3C7"
    return               "🔴 Major Delay",   "#991B1B", "#FEE2E2"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='color:#1A3A5C; margin-bottom:0'>🚌 BMTC Delay Predictor</h1>
    <p style='color:gray; margin-top:4px'>
        Bengaluru · ML-Powered Bus Delay Forecasting ·
    </p>
    <hr>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Predict Journey",
    "📊 Model Results",
    "🧠 Delay Classifier",
    "🛠️ Admin Dashboard",
    "ℹ️ About Project",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — JOURNEY PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Your Journey Details")

    # ── Stop inputs ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        from_input = st.text_input("📍 FROM Stop",
                                    placeholder="e.g. Halasuru, Hebbal")
    with col2:
        to_input   = st.text_input("🏁 TO Stop",
                                    placeholder="e.g. Majestic, Silk Board")

    # ── Calendar date picker ──────────────────────────────────────────────────
    st.markdown("📅 **Select Travel Date**")
    today       = date.today()
    max_date    = today + timedelta(days=30)
    travel_date = st.date_input(
        " ",
        value=today,
        min_value=today,
        max_value=max_date,
        label_visibility="collapsed"
    )
    dow   = travel_date.weekday()
    month = travel_date.month
    day   = ["Monday","Tuesday","Wednesday","Thursday",
             "Friday","Saturday","Sunday"][dow]
    st.caption(
        f"📆 {travel_date.strftime('%d %B %Y')}  ({day})  "
        f"— Month: {travel_date.strftime('%B')}"
    )

    # ── Time selector — 30-min slots, defaults to current IST time ───────────
    st.markdown("⏰ **Time of Travel**")

    now_ist      = get_ist_now()
    default_idx  = slot_index_for(now_ist.hour, now_ist.minute)

    # Show current IST time as a hint above the selector
    st.caption(
        f"🕐 Current Bengaluru time: **{now_ist.strftime('%H:%M IST')}** "
        f"— defaulting to nearest 30-min slot"
    )

    selected_label = st.selectbox(
        " ",
        options=TIME_LABELS,
        index=default_idx,
        label_visibility="collapsed",
        help="Select your departure time. Defaults to current Bengaluru time."
    )

    # Parse the selected slot back to hour + minute
    hour, minute = TIME_VALUES[TIME_LABELS.index(selected_label)]

    # Show rush-hour / time-period context label
    if hour in [7, 8, 9]:
        st.caption("🔴 AM Rush Hour — expect higher delays")
    elif hour in [17, 18, 19]:
        st.caption("🔴 PM Rush Hour — expect higher delays")
    elif 0 <= hour <= 5:
        st.caption("🌙 Late Night — minimal traffic expected")
    else:
        st.caption("🟡 Normal Hours")

    # ── Live weather from OpenWeatherMap ──────────────────────────────────────
    st.markdown("🌦️ **Weather Conditions**")

    live_rain, live_desc, live_temp, live_icon = fetch_bengaluru_weather()

    if live_rain is not None:
        # API call succeeded — show live weather and use it as default
        wcol1, wcol2 = st.columns([1, 4])
        with wcol1:
            if live_icon:
                st.image(live_icon, width=60)
        with wcol2:
            rain_emoji = "🌧️" if live_rain else "☀️"
            st.markdown(
                f"**Live Bengaluru weather:** {rain_emoji} {live_desc}  "
                f"| 🌡️ {live_temp}°C"
            )
            st.caption("Auto-detected via OpenWeatherMap · updates every 10 min")

        # Toggle defaults to live rain state but user can still override
        is_rain = int(st.toggle(
            "🌧️ Raining? (auto-detected — override if needed)",
            value=bool(live_rain)
        ))

    elif live_rain is None and live_desc == "no_key":
        # No API key configured — fall back to manual toggle
        st.info(
            "💡 Add your free OpenWeatherMap API key to Streamlit secrets "
            "for live rain detection. Using manual toggle for now.",
            icon="ℹ️"
        )
        is_rain = int(st.toggle("🌧️ Raining?", value=False))

    else:
        # API key present but call failed (network issue, bad key, etc.)
        st.warning(f"Weather fetch failed ({live_desc}). Using manual toggle.")
        is_rain = int(st.toggle("🌧️ Raining?", value=False))

    if month in [6, 7, 8, 9]:
        st.caption("☔ Monsoon season — rain likely to add 2–5 min extra delay")

    # ── Stop search dropdowns ─────────────────────────────────────────────────
    src_stop, dst_stop = None, None

    if from_input:
        from_matches = find_stop(from_input)
        if from_matches:
            src_stop = st.selectbox("Select FROM stop:", from_matches, key="src")
        else:
            st.warning("No FROM stop found. Try a different name.")

    if to_input:
        to_matches = find_stop(to_input)
        if to_matches:
            dst_stop = st.selectbox("Select TO stop:", to_matches, key="dst")
        else:
            st.warning("No TO stop found. Try a different name.")

    # ── Predict button ────────────────────────────────────────────────────────
    if st.button("🔍 Predict Delay", type="primary", use_container_width=True):

        if not src_stop or not dst_stop:
            st.error("Please enter both FROM and TO stops.")
        elif src_stop == dst_stop:
            st.error("FROM and TO stops cannot be the same.")
        else:
            # ── Run delay prediction + bus lookup in parallel ─────────────────
            src_delay = predict_delay(src_stop, hour, dow, month, is_rain)
            dst_delay = predict_delay(dst_stop, hour, dow, month, is_rain)

            src_label, src_color, src_bg = get_status(src_delay)
            dst_label, dst_color, dst_bg = get_status(dst_delay)
            worse = max(src_delay, dst_delay)

            with st.spinner("🔍 Looking up bus numbers..."):
                bus_result = find_buses(src_stop, dst_stop)

            buses  = bus_result["buses"]
            source = bus_result["source"]
            note   = bus_result["note"]

            # ── Sanitise note: strip any markdown link syntax ─────────────────
            note = re.sub(
                r"\[([^\]]+)\]\((https?://[^\)]+)\)",
                r"\1 (\2)",
                note
            )

            # ── Weather & time meta ───────────────────────────────────────────
            weather_icon = "🌧️ Rain" if is_rain else "☀️ Clear"
            rush_note    = " · 🔴 AM Rush" if hour in [7,8,9] else (
                           " · 🔴 PM Rush" if hour in [17,18,19] else "")

            # ── Tip ──────────────────────────────────────────────────────────
            if worse >= 8:
                tip_bg, tip_icon, tip_text = "#FEE2E2", "🔴", "Leave early — heavy delays expected!"
            elif worse >= 3:
                tip_bg, tip_icon, tip_text = "#FEF3C7", "⚠️", "Keep a 10-minute buffer for this journey."
            else:
                tip_bg, tip_icon, tip_text = "#D1FAE5", "✅", "Good time to travel — minimal delays expected."

            # ── Source badge ─────────────────────────────────────────────────
            dot_color, dot_label = {
                "live"  : ("#22C55E", "🟢 Live data"),
                "gtfs"  : ("#16A34A", "🟢 BMTC GTFS data"),
                "static": ("#F97316", "🟠 Estimated"),
                "none"  : ("#9CA3AF", "⚪ Not found"),
            }.get(source, ("#9CA3AF", ""))

            # ── Bus pills HTML (no f-string quotes inside style) ─────────────
            PILL = (
                "display:inline-block;"
                "background:#1E3A5F;"
                "color:#E0F2FE;"
                "border-radius:6px;"
                "padding:4px 10px;"
                "margin:3px 4px 3px 0;"
                "font-weight:700;"
                "font-size:0.9rem;"
                "letter-spacing:0.02em"
            )
            if buses:
                pills = "".join(f'<span style="{PILL}">{b}</span>' for b in buses)
                bus_html = (
                    '<div style="margin-top:14px;padding-top:14px;'
                    'border-top:1px solid #E2E8F0">'
                    '<p style="margin:0 0 8px 0;font-weight:700;color:#1A3A5C;font-size:0.9rem">'
                    "🚌 Buses on this corridor</p>"
                    f'<div style="line-height:2">{pills}</div>'
                    f'<p style="margin:8px 0 0 0;font-size:0.75rem;color:{dot_color}">'
                    f"{dot_label} · {note}</p>"
                    "</div>"
                )
            else:
                bus_html = (
                    '<div style="margin-top:14px;padding-top:14px;'
                    'border-top:1px solid #E2E8F0">'
                    '<p style="margin:0;font-size:0.85rem;color:#6B7280">'
                    f"🚌 Bus numbers unavailable for this corridor.<br>"
                    f'<span style="font-size:0.75rem">{note}</span></p>'
                    "</div>"
                )

            # ══════════════════════════════════════════════════════════════════
            # UNIFIED OUTPUT CARD — rendered via st.components.v1.html()
            # so Streamlit's markdown parser never touches the HTML
            # ══════════════════════════════════════════════════════════════════
            import streamlit.components.v1 as components

            card_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ margin:0; padding:0; font-family: -apple-system, BlinkMacSystemFont,
          'Segoe UI', Roboto, sans-serif; background:transparent; }}
  .card {{
    background:#FFFFFF;
    border:1px solid #E2E8F0;
    border-radius:14px;
    padding:22px 24px 18px 24px;
    box-shadow:0 2px 12px rgba(0,0,0,0.07);
  }}
  .meta {{ font-size:0.8rem; color:#64748B; margin:4px 0 16px 0; }}
  .route-header {{ display:flex; align-items:center; gap:8px; margin-bottom:4px; }}
  .route-header span {{ font-size:1.05rem; font-weight:700; color:#1A3A5C; }}
  .route-header .arrow {{ color:#64748B; font-size:1.1rem; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
  .delay-card {{
    border-radius:10px;
    padding:16px 18px;
  }}
  .delay-card .role {{
    margin:0 0 2px 0;
    font-size:0.72rem;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:0.07em;
  }}
  .delay-card .stop-name {{
    margin:0 0 6px 0;
    font-size:0.85rem;
    font-weight:600;
  }}
  .delay-card .delay-val {{
    margin:0;
    font-size:2rem;
    font-weight:800;
    line-height:1;
  }}
  .delay-card .delay-val span {{ font-size:1rem; font-weight:500; }}
  .delay-card .status {{
    margin:4px 0 0 0;
    font-size:0.85rem;
  }}
  .tip {{
    margin-top:14px;
    border-radius:8px;
    padding:10px 14px;
    font-size:0.88rem;
    font-weight:600;
    color:#1A3A5C;
  }}
</style>
</head>
<body>
<div class="card">

  <div class="route-header">
    <span>📍 {src_stop}</span>
    <span class="arrow">→</span>
    <span>🏁 {dst_stop}</span>
  </div>
  <p class="meta">
    📅 {travel_date.strftime('%d %b %Y')} ({day})
    &nbsp;·&nbsp; ⏰ {selected_label}
    &nbsp;·&nbsp; {weather_icon}{rush_note}
  </p>

  <div class="grid">
    <div class="delay-card"
         style="background:{src_bg};border-left:5px solid {src_color}">
      <p class="role"    style="color:{src_color}">FROM</p>
      <p class="stop-name" style="color:{src_color}">{src_stop}</p>
      <p class="delay-val" style="color:{src_color}">{src_delay}
         <span>min</span></p>
      <p class="status"  style="color:{src_color}">{src_label}</p>
    </div>
    <div class="delay-card"
         style="background:{dst_bg};border-left:5px solid {dst_color}">
      <p class="role"    style="color:{dst_color}">TO</p>
      <p class="stop-name" style="color:{dst_color}">{dst_stop}</p>
      <p class="delay-val" style="color:{dst_color}">{dst_delay}
         <span>min</span></p>
      <p class="status"  style="color:{dst_color}">{dst_label}</p>
    </div>
  </div>

  {bus_html}

  <div class="tip" style="background:{tip_bg}">
    {tip_icon} {tip_text}
  </div>

</div>
</body>
</html>
"""
            st.markdown("---")
            components.html(card_html, height=420, scrolling=False)

            # ── 24-hour forecast chart ────────────────────────────────────────
            st.markdown("#### 24-Hour Delay Forecast")

            src_has_prophet = src_stop in prophet_stops
            dst_has_prophet = dst_stop in prophet_stops
            hours           = list(range(24))

            if src_has_prophet or dst_has_prophet:
                def get_24h(stop_name, has_prophet):
                    if has_prophet:
                        m = load_prophet_model(stop_name)
                        if m:
                            future = pd.DataFrame({
                                "ds"     : pd.date_range("2024-07-01",
                                                          periods=24, freq="h"),
                                "is_rush": [1 if h in [7,8,9,17,18,19] else 0
                                            for h in range(24)]
                            })
                            fc = m.predict(future)
                            return (fc["yhat"].clip(lower=0).tolist(),
                                    fc["yhat_lower"].clip(lower=0).tolist(),
                                    fc["yhat_upper"].clip(lower=0).tolist())
                    vals = [predict_delay(stop_name, h, dow, month, is_rain)
                            for h in hours]
                    return vals, None, None

                src_24, src_lo, src_hi = get_24h(src_stop, src_has_prophet)
                dst_24, dst_lo, dst_hi = get_24h(dst_stop, dst_has_prophet)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(hours, src_24,
                        label=f"{src_stop[:22]} "
                              f"{'(Prophet)' if src_has_prophet else '(XGBoost)'}",
                        color='steelblue', lw=2, marker='o', markersize=3)
                if src_lo:
                    ax.fill_between(hours, src_lo, src_hi,
                                    alpha=0.12, color='steelblue')
                ax.plot(hours, dst_24,
                        label=f"{dst_stop[:22]} "
                              f"{'(Prophet)' if dst_has_prophet else '(XGBoost)'}",
                        color='crimson', lw=2, marker='s', markersize=3)
                if dst_lo:
                    ax.fill_between(hours, dst_lo, dst_hi,
                                    alpha=0.12, color='crimson')
            else:
                src_24 = [predict_delay(src_stop, h, dow, month, is_rain)
                          for h in hours]
                dst_24 = [predict_delay(dst_stop, h, dow, month, is_rain)
                          for h in hours]
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(hours, src_24,
                        label=f"{src_stop[:25]} (XGBoost)",
                        color='steelblue', lw=2, marker='o', markersize=3)
                ax.plot(hours, dst_24,
                        label=f"{dst_stop[:25]} (XGBoost)",
                        color='crimson', lw=2, marker='s', markersize=3)

            # Mark selected time on the chart
            ax.axvline(hour + minute/60, color='gray', ls='--', lw=1.5,
                       label=f'Your time ({selected_label})')
            ax.axvspan(7,  9,  alpha=0.10, color='red',    label='AM Rush')
            ax.axvspan(17, 19, alpha=0.10, color='orange', label='PM Rush')
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Predicted Delay (min)")
            ax.set_title(
                f"{travel_date.strftime('%d %b %Y')} ({day})"
                f"{'  🌧️ Rain' if is_rain else ''}"
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_xticks(range(0, 24, 2))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            if src_has_prophet or dst_has_prophet:
                st.caption(
                    "Shaded band = 80% confidence interval (Prophet stops). "
                    "Stops without a Prophet model use XGBoost point estimate."
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance Comparison")

    n_stops = metadata.get("n_stops", "~1,955")
    st.markdown(
        f"Models trained on 180 days of hourly delay data across "
        f"**{n_stops:,} BMTC stops**. "
        "Evaluated on a time-based held-out 20% test set "
        "(future dates the model never saw)."
    )

    st.dataframe(
        final_results.set_index("Model"),
        use_container_width=True
    )
    st.caption(
        "RMSE and MAE in minutes — lower is better. "
        "XGBoost (all stops) is the primary model used for live prediction. "
        "ARIMA/SARIMA/LSTM metrics are from the busiest stop only."
    )

    st.markdown("#### RMSE & MAE — All Models")
    img = os.path.join(OUTPUT_DIR, "model_comparison.png")
    if os.path.exists(img):
        st.image(img, use_container_width=True)
    else:
        plot_df = final_results.copy()
        plot_df["RMSE"] = pd.to_numeric(plot_df["RMSE"], errors="coerce")
        plot_df["MAE"]  = pd.to_numeric(plot_df["MAE"],  errors="coerce")
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(plot_df))
        ax.bar(x - 0.2, plot_df["RMSE"], 0.35, label="RMSE", color="steelblue")
        ax.bar(x + 0.2, plot_df["MAE"],  0.35, label="MAE",  color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["Model"], rotation=12, ha="right", fontsize=8)
        ax.set_ylabel("Error (minutes)")
        ax.set_title("RMSE & MAE — All Models")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    image_sections = [
        ("XGBoost Feature Importance",           "feature_importance.png"),
        ("XGBoost Training Curve",               "xgb_training_curve.png"),
        ("XGBoost — Actual vs Predicted",        "xgb_predictions.png"),
        ("Prophet 24-Hour Forecasts (Top 6)",    "prophet_forecasts.png"),
        ("ARIMA vs SARIMA — Actual vs Predicted","arima_sarima.png"),
        ("LSTM — Training Curves",               "lstm_training.png"),
        ("LSTM — Actual vs Predicted",           "lstm_predictions.png"),
        ("EDA — Delay Patterns",                 "eda.png"),
        ("Rain Effect on Delays",                "rain_effect.png"),
    ]
    for title, fname in image_sections:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            st.markdown(f"#### {title}")
            st.image(path, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DELAY CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🧠 Delay Severity Classifier")
    st.markdown(
        "Classifies a stop's delay into **On-Time / Minor Delay / Major Delay** "
        "using the XGBoost regression output + domain thresholds. "
        "Shows predicted probability per class and the top feature drivers."
    )
    st.markdown("---")

    # ── Inputs ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        cls_stop_input = st.text_input("🔍 Stop Name", placeholder="e.g. Hebbal, Silk Board",
                                       key="cls_stop")
    with c2:
        cls_hour = st.slider("⏰ Hour of Day", 0, 23,
                             value=get_ist_now().hour, key="cls_hour")

    c3, c4, c5 = st.columns(3)
    with c3:
        cls_dow = st.selectbox("📅 Day", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                               key="cls_dow")
        dow_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
        cls_dow_int = dow_map[cls_dow]
    with c4:
        cls_month = st.selectbox("📆 Month", list(range(1,13)),
                                 format_func=lambda m: datetime(2024,m,1).strftime("%B"),
                                 index=get_ist_now().month - 1, key="cls_month")
    with c5:
        cls_rain = int(st.toggle("🌧️ Rain", key="cls_rain"))

    cls_matches = find_stop(cls_stop_input) if cls_stop_input else []
    cls_stop = None
    if cls_matches:
        cls_stop = st.selectbox("Select stop:", cls_matches, key="cls_sel")
    elif cls_stop_input:
        st.warning("Stop not found — try a different name.")

    if st.button("🧠 Classify Delay", type="primary", use_container_width=True, key="cls_btn"):
        if not cls_stop:
            st.error("Please select a valid stop first.")
        else:
            X_cls, s_cls = build_features(cls_stop, cls_hour, cls_dow_int, cls_month, cls_rain)
            if X_cls is None:
                st.error("Feature build failed — stop not in stop_summary.")
            else:
                raw_delay = float(np.clip(xgb_model.predict(X_cls)[0], 0, None))

                # ── Classify ──────────────────────────────────────────────────
                # Thresholds: <3 min = On-Time, 3–8 = Minor, ≥8 = Major
                if raw_delay < 3:
                    cls_label = "✅ On-Time"
                    cls_color = "#065F46"; cls_bg = "#D1FAE5"
                    # Soft-probability: triangle distribution around 0
                    p_ontime = max(0.0, min(1.0, 1.0 - raw_delay / 3.0))
                    p_minor  = 1.0 - p_ontime
                    p_major  = 0.0
                elif raw_delay < 8:
                    cls_label = "⚠️ Minor Delay"
                    cls_color = "#92400E"; cls_bg = "#FEF3C7"
                    t = (raw_delay - 3) / 5.0          # 0→1 across [3,8]
                    p_minor  = max(0.4, 1.0 - abs(t - 0.5))
                    p_ontime = max(0.0, 0.5 - t * 0.5)
                    p_major  = max(0.0, t * 0.5)
                    total    = p_ontime + p_minor + p_major
                    p_ontime /= total; p_minor /= total; p_major /= total
                else:
                    cls_label = "🔴 Major Delay"
                    cls_color = "#991B1B"; cls_bg = "#FEE2E2"
                    p_major  = min(1.0, 0.5 + (raw_delay - 8) / 20.0)
                    p_minor  = 1.0 - p_major
                    p_ontime = 0.0

                # ── Feature importance / impact ────────────────────────────────
                feat_vals = X_cls.iloc[0].to_dict()
                # Rank features by absolute value × model feature importance
                try:
                    fi = dict(zip(FEATURES, xgb_model.feature_importances_))
                    impact = {f: abs(feat_vals.get(f, 0)) * fi.get(f, 0) for f in FEATURES}
                    top_feats = sorted(impact.items(), key=lambda x: x[1], reverse=True)[:6]
                except Exception:
                    top_feats = []

                # ── Render result ─────────────────────────────────────────────
                import streamlit.components.v1 as components

                prob_bar = lambda p, col, lbl: (
                    f'<div style="margin:6px 0">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.8rem;margin-bottom:2px">'
                    f'<span>{lbl}</span><span>{p*100:.1f}%</span></div>'
                    f'<div style="background:#E5E7EB;border-radius:4px;height:10px">'
                    f'<div style="background:{col};width:{p*100:.1f}%;height:10px;'
                    f'border-radius:4px"></div></div></div>'
                )

                feat_rows = ""
                feat_colors = {
                    "is_rain":"#3B82F6","is_rush":"#EF4444","hour":"#F59E0B",
                    "factor":"#8B5CF6","trip_count":"#06B6D4","route_count":"#10B981",
                }
                for fn, fv in top_feats:
                    disp = fn.replace("_"," ").title()
                    raw_v = feat_vals.get(fn, 0)
                    fc = feat_colors.get(fn, "#6B7280")
                    feat_rows += (
                        f'<tr><td style="padding:4px 8px;font-size:0.8rem">{disp}</td>'
                        f'<td style="padding:4px 8px;font-size:0.8rem;text-align:right">'
                        f'<span style="background:{fc}22;color:{fc};padding:2px 6px;'
                        f'border-radius:4px;font-weight:600">{raw_v:.3f}</span></td></tr>'
                    )

                rush_tag = ""
                if cls_hour in [7,8,9]:   rush_tag = "🔴 AM Rush"
                elif cls_hour in [17,18,19]: rush_tag = "🔴 PM Rush"
                rain_tag = "🌧️ Rain" if cls_rain else "☀️ Clear"

                html = f"""
<html><head><meta charset="utf-8">
<style>
  body{{margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}
  .wrap{{background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:22px 24px;
          box-shadow:0 2px 12px rgba(0,0,0,.07)}}
  .badge{{display:inline-block;padding:6px 16px;border-radius:8px;
           font-weight:800;font-size:1.1rem;margin-bottom:12px}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px}}
  .box{{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:14px}}
  .box h4{{margin:0 0 10px 0;font-size:0.85rem;color:#64748B;font-weight:700;
            text-transform:uppercase;letter-spacing:.06em}}
  table{{width:100%;border-collapse:collapse}}
</style></head><body>
<div class="wrap">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
    <div class="badge" style="background:{cls_bg};color:{cls_color}">{cls_label}</div>
    <span style="font-size:0.85rem;color:#64748B">
      {cls_stop} &nbsp;·&nbsp; {cls_hour:02d}:00 &nbsp;·&nbsp; {cls_dow} &nbsp;·&nbsp;
      {datetime(2024,cls_month,1).strftime('%B')} &nbsp;·&nbsp; {rain_tag}
      {"&nbsp;·&nbsp;" + rush_tag if rush_tag else ""}
    </span>
  </div>
  <p style="font-size:1.6rem;font-weight:800;color:{cls_color};margin:8px 0 0 0">
    {raw_delay:.1f} min <span style="font-size:1rem;font-weight:400">predicted delay</span>
  </p>

  <div class="grid">
    <div class="box">
      <h4>Class Probabilities</h4>
      {prob_bar(p_ontime,"#10B981","✅ On-Time")}
      {prob_bar(p_minor, "#F59E0B","⚠️ Minor Delay")}
      {prob_bar(p_major, "#EF4444","🔴 Major Delay")}
    </div>
    <div class="box">
      <h4>Top Feature Drivers</h4>
      <table>{feat_rows}</table>
    </div>
  </div>
</div>
</body></html>"""
                components.html(html, height=310, scrolling=False)

                # ── Multi-stop batch classification ───────────────────────────
                st.markdown("#### Compare Multiple Stops at This Time")
                st.caption("Top 10 most-delayed vs top 10 best-performing stops right now")

                sample_stops = stop_summary.nlargest(10,"avg_delay")["stop_name"].tolist() + \
                               stop_summary.nsmallest(10,"avg_delay")["stop_name"].tolist()
                batch_rows = []
                for sn in sample_stops:
                    d = predict_delay(sn, cls_hour, cls_dow_int, cls_month, cls_rain)
                    lbl = "On-Time" if d < 3 else ("Minor Delay" if d < 8 else "Major Delay")
                    batch_rows.append({"Stop": sn, "Predicted Delay (min)": d, "Class": lbl})

                bdf = pd.DataFrame(batch_rows).sort_values("Predicted Delay (min)", ascending=False)

                fig_b, ax_b = plt.subplots(figsize=(10, 5))
                colors_b = ["#EF4444" if c=="Major Delay" else
                            "#F59E0B" if c=="Minor Delay" else "#10B981"
                            for c in bdf["Class"]]
                ax_b.barh(bdf["Stop"].str[:30], bdf["Predicted Delay (min)"],
                          color=colors_b, edgecolor="white")
                ax_b.axvline(3, color="#F59E0B", ls="--", lw=1.2, label="Minor threshold (3 min)")
                ax_b.axvline(8, color="#EF4444", ls="--", lw=1.2, label="Major threshold (8 min)")
                ax_b.set_xlabel("Predicted Delay (min)")
                ax_b.set_title(f"Delay Classification — {cls_hour:02d}:00 · {cls_dow} · "
                               f"{'Rain' if cls_rain else 'Clear'}")
                ax_b.legend(fontsize=8)
                ax_b.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_b)
                plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ADMIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    import streamlit.components.v1 as components

    st.subheader("🛠️ Admin Dashboard — Network Overview")
    st.markdown(
        "System-wide analytics derived from the trained model and stop metadata. "
        "No external data source required — all computed on-the-fly."
    )
    st.markdown("---")

    # ── Admin password gate ───────────────────────────────────────────────────
    _ADMIN_PW = "bmtc2024"
    if "admin_unlocked" not in st.session_state:
        st.session_state["admin_unlocked"] = False

    if not st.session_state["admin_unlocked"]:
        pw = st.text_input("🔒 Enter Admin Password", type="password", key="admin_pw")
        if st.button("Unlock Dashboard", key="admin_unlock_btn"):
            if pw == _ADMIN_PW:
                st.session_state["admin_unlocked"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.caption("Default password: `bmtc2024`  — change `_ADMIN_PW` in app.py")
        st.stop()

    # ── Controls ──────────────────────────────────────────────────────────────
    adm_col1, adm_col2, adm_col3 = st.columns(3)
    with adm_col1:
        adm_hour  = st.slider("⏰ Analysis Hour", 0, 23, value=8, key="adm_h")
    with adm_col2:
        adm_dow   = st.selectbox("📅 Day", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                                  key="adm_dow")
        adm_dow_i = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}[adm_dow]
    with adm_col3:
        adm_rain  = int(st.toggle("🌧️ Rain scenario", key="adm_rain"))

    adm_month = get_ist_now().month   # always current month

    # ── Compute network-wide predictions ─────────────────────────────────────
    @st.cache_data(ttl=300, show_spinner=False)
    def _network_predictions(hour: int, dow: int, month: int, rain: int) -> pd.DataFrame:
        rows = []
        for _, row in stop_summary.iterrows():
            sn = row["stop_name"]
            d  = predict_delay(sn, hour, dow, month, rain)
            lbl = "On-Time" if d < 3 else ("Minor" if d < 8 else "Major")
            rows.append({
                "stop_name"   : sn,
                "delay_min"   : d,
                "class"       : lbl,
                "avg_delay"   : float(row["avg_delay"]),
                "trip_count"  : int(row["trip_count"]),
                "route_count" : int(row["route_count"]),
            })
        return pd.DataFrame(rows)

    with st.spinner("Computing network predictions..."):
        net = _network_predictions(adm_hour, adm_dow_i, adm_month, adm_rain)

    n_total  = len(net)
    n_ontime = (net["class"] == "On-Time").sum()
    n_minor  = (net["class"] == "Minor").sum()
    n_major  = (net["class"] == "Major").sum()
    avg_net  = net["delay_min"].mean()
    max_net  = net["delay_min"].max()

    # ── KPI row ───────────────────────────────────────────────────────────────
    kpi_html = f"""
<html><head><meta charset="utf-8">
<style>
  body{{margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}
  .row{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}}
  .kpi{{background:#fff;border:1px solid #E2E8F0;border-radius:12px;
         padding:16px 18px;box-shadow:0 1px 6px rgba(0,0,0,.05);text-align:center}}
  .kpi .val{{font-size:1.9rem;font-weight:800;line-height:1}}
  .kpi .lbl{{font-size:0.72rem;color:#64748B;margin-top:4px;font-weight:600;
              text-transform:uppercase;letter-spacing:.05em}}
</style></head><body>
<div class="row">
  <div class="kpi">
    <div class="val" style="color:#1A3A5C">{n_total:,}</div>
    <div class="lbl">Total Stops</div>
  </div>
  <div class="kpi">
    <div class="val" style="color:#10B981">{n_ontime:,}</div>
    <div class="lbl">✅ On-Time</div>
  </div>
  <div class="kpi">
    <div class="val" style="color:#F59E0B">{n_minor:,}</div>
    <div class="lbl">⚠️ Minor Delay</div>
  </div>
  <div class="kpi">
    <div class="val" style="color:#EF4444">{n_major:,}</div>
    <div class="lbl">🔴 Major Delay</div>
  </div>
  <div class="kpi">
    <div class="val" style="color:#6366F1">{avg_net:.1f} min</div>
    <div class="lbl">Avg Network Delay</div>
  </div>
</div>
</body></html>"""
    components.html(kpi_html, height=110)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Delay distribution + class pie ─────────────────────────────────
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("#### Delay Distribution")
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.hist(net["delay_min"], bins=40, color="#3B82F6", edgecolor="white", alpha=0.85)
        ax1.axvline(3, color="#F59E0B", ls="--", lw=1.5, label="Minor (3 min)")
        ax1.axvline(8, color="#EF4444", ls="--", lw=1.5, label="Major (8 min)")
        ax1.set_xlabel("Predicted Delay (min)")
        ax1.set_ylabel("Number of Stops")
        ax1.set_title(f"{adm_hour:02d}:00 · {adm_dow} · {'Rain' if adm_rain else 'Clear'}")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1); plt.close()

    with r1c2:
        st.markdown("#### Network Status Breakdown")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sizes  = [n_ontime, n_minor, n_major]
        labels = [f"On-Time\n{n_ontime}", f"Minor\n{n_minor}", f"Major\n{n_major}"]
        colors = ["#10B981", "#F59E0B", "#EF4444"]
        explode = [0.04, 0.04, 0.08]
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=labels, colors=colors, explode=explode,
            autopct="%1.1f%%", startangle=120,
            textprops={"fontsize": 8}, pctdistance=0.78,
        )
        for at in autotexts:
            at.set_fontsize(8); at.set_color("white"); at.set_fontweight("bold")
        ax2.set_title("Status Distribution", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    # ── Row 2: Worst / Best stops ─────────────────────────────────────────────
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("#### 🔴 Top 15 Worst Stops")
        worst = net.nlargest(15, "delay_min")[["stop_name", "delay_min", "class"]]
        fig3, ax3 = plt.subplots(figsize=(5, 4.5))
        bar_colors = ["#EF4444" if c == "Major" else "#F59E0B" for c in worst["class"]]
        ax3.barh(worst["stop_name"].str[:28], worst["delay_min"],
                 color=bar_colors, edgecolor="white")
        ax3.set_xlabel("Predicted Delay (min)")
        ax3.set_title("Worst Stops — Current Scenario")
        ax3.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3); plt.close()

    with r2c2:
        st.markdown("#### ✅ Top 15 Best Stops")
        best = net.nsmallest(15, "delay_min")[["stop_name", "delay_min", "class"]]
        fig4, ax4 = plt.subplots(figsize=(5, 4.5))
        ax4.barh(best["stop_name"].str[:28], best["delay_min"],
                 color="#10B981", edgecolor="white")
        ax4.set_xlabel("Predicted Delay (min)")
        ax4.set_title("Best Stops — Current Scenario")
        ax4.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4); plt.close()

    # ── Row 3: 24-hr heatmap by hour & class ──────────────────────────────────
    st.markdown("#### ⏱️ Network-Wide Hourly Delay Heatmap")
    st.caption("Average predicted delay per hour across all stops · current day/month/rain setting")

    @st.cache_data(ttl=600, show_spinner=False)
    def _hourly_heatmap(dow: int, month: int, rain: int) -> np.ndarray:
        """Returns shape (n_stops, 24) matrix of predicted delays."""
        stops_list = stop_summary["stop_name"].tolist()
        mat = np.zeros((len(stops_list), 24))
        for h in range(24):
            for i, sn in enumerate(stops_list):
                mat[i, h] = predict_delay(sn, h, dow, month, rain)
        return mat, stops_list

    with st.spinner("Building hourly heatmap (runs once per scenario)..."):
        mat, stops_list = _hourly_heatmap(adm_dow_i, adm_month, adm_rain)

    hourly_avg = mat.mean(axis=0)   # shape (24,)
    hourly_pct_major = (mat >= 8).mean(axis=0) * 100  # % stops in major delay per hour

    fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax5a.plot(range(24), hourly_avg, color="#3B82F6", lw=2.5, marker="o", markersize=4)
    ax5a.axhline(3, color="#F59E0B", ls="--", lw=1.2)
    ax5a.axhline(8, color="#EF4444", ls="--", lw=1.2)
    ax5a.axvspan(7, 9,  alpha=0.10, color="red")
    ax5a.axvspan(17, 19, alpha=0.10, color="orange")
    ax5a.set_ylabel("Avg Delay (min)")
    ax5a.set_title(f"Network Hourly Profile — {adm_dow} · "
                   f"{'Rain' if adm_rain else 'Clear'}")
    ax5a.grid(alpha=0.3)

    ax5b.fill_between(range(24), hourly_pct_major, color="#EF4444", alpha=0.5)
    ax5b.plot(range(24), hourly_pct_major, color="#EF4444", lw=1.5)
    ax5b.set_ylabel("% Stops Major Delay")
    ax5b.set_xlabel("Hour of Day")
    ax5b.set_xticks(range(0, 24, 2))
    ax5b.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig5); plt.close()

    # ── Row 4: Rain impact analysis ───────────────────────────────────────────
    st.markdown("#### 🌧️ Rain Impact Analysis")
    st.caption("Comparing dry vs rainy scenario for the selected hour across all stops")

    @st.cache_data(ttl=600, show_spinner=False)
    def _rain_delta(hour: int, dow: int, month: int) -> pd.DataFrame:
        rows = []
        for _, row in stop_summary.iterrows():
            sn   = row["stop_name"]
            d_dry  = predict_delay(sn, hour, dow, month, 0)
            d_rain = predict_delay(sn, hour, dow, month, 1)
            rows.append({"stop": sn, "dry": d_dry, "rain": d_rain,
                         "delta": d_rain - d_dry,
                         "trip_count": int(row["trip_count"])})
        return pd.DataFrame(rows)

    with st.spinner("Computing rain delta..."):
        rain_df = _rain_delta(adm_hour, adm_dow_i, adm_month)

    rc1, rc2 = st.columns(2)
    with rc1:
        avg_dry  = rain_df["dry"].mean()
        avg_rain = rain_df["rain"].mean()
        fig6, ax6 = plt.subplots(figsize=(5, 3))
        ax6.bar(["☀️ Dry", "🌧️ Rain"], [avg_dry, avg_rain],
                color=["#3B82F6", "#6366F1"], edgecolor="white", width=0.5)
        ax6.set_ylabel("Avg Delay (min)")
        ax6.set_title(f"Rain Effect · {adm_hour:02d}:00 · {adm_dow}")
        for i, v in enumerate([avg_dry, avg_rain]):
            ax6.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax6.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig6); plt.close()

    with rc2:
        top_rain_impact = rain_df.nlargest(10, "delta")[["stop", "delta"]]
        fig7, ax7 = plt.subplots(figsize=(5, 3))
        ax7.barh(top_rain_impact["stop"].str[:28], top_rain_impact["delta"],
                 color="#6366F1", edgecolor="white")
        ax7.set_xlabel("Extra Delay in Rain (min)")
        ax7.set_title("Stops Most Affected by Rain")
        ax7.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig7); plt.close()

    # ── Row 5: Stop Audit Table ────────────────────────────────────────────────
    st.markdown("#### 🔎 Full Stop Audit")
    audit_search = st.text_input("Filter stops by name:", key="audit_search")
    audit_df = net.copy()
    audit_df.columns = ["Stop Name", "Predicted Delay (min)", "Severity Class",
                        "Hist. Avg Delay", "Trip Count", "Route Count"]
    if audit_search:
        audit_df = audit_df[audit_df["Stop Name"].str.contains(
            audit_search, case=False, na=False)]

    # Colour-code the class column
    def _highlight_class(val):
        if val == "Major":   return "background-color:#FEE2E2;color:#991B1B;font-weight:700"
        if val == "Minor":   return "background-color:#FEF3C7;color:#92400E;font-weight:700"
        return "background-color:#D1FAE5;color:#065F46;font-weight:700"

    styled = (
        audit_df.sort_values("Predicted Delay (min)", ascending=False)
        .reset_index(drop=True)
        .style
        .applymap(_highlight_class, subset=["Severity Class"])
        .format({"Predicted Delay (min)": "{:.1f}", "Hist. Avg Delay": "{:.2f}"})
    )
    st.dataframe(styled, use_container_width=True, height=400)

    csv_bytes = audit_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Audit CSV",
        data=csv_bytes,
        file_name=f"bmtc_audit_{adm_dow}_{adm_hour:02d}h_"
                  f"{'rain' if adm_rain else 'dry'}.csv",
        mime="text/csv",
        key="audit_dl",
    )

    # ── Row 6: Scenario Comparison ─────────────────────────────────────────────
    st.markdown("#### 📊 Scenario Comparison (Rush Hour vs Off-Peak)")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.caption("AM Rush — 08:00 Monday, Rain")
        rush_preds = [predict_delay(s, 8, 0, adm_month, 1)
                      for s in stop_summary["stop_name"]]
        st.metric("Avg Delay", f"{np.mean(rush_preds):.2f} min")
        st.metric("% Major Delay Stops",
                  f"{100*sum(d>=8 for d in rush_preds)/len(rush_preds):.1f}%")
    with sc2:
        st.caption("Off-Peak — 14:00 Wednesday, Clear")
        offpeak_preds = [predict_delay(s, 14, 2, adm_month, 0)
                         for s in stop_summary["stop_name"]]
        st.metric("Avg Delay", f"{np.mean(offpeak_preds):.2f} min")
        st.metric("% Major Delay Stops",
                  f"{100*sum(d>=8 for d in offpeak_preds)/len(offpeak_preds):.1f}%")



# TAB 4 — PROFILE
# ==========================================================
with tab4:
    st.subheader("👤 User Profile")

    st.write(f"Logged in as: {st.session_state.user}")

    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()




# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("About This Project")

    n_stops   = metadata.get("n_stops", "~1,955")
    n_prophet = len(prophet_stops)

    st.markdown(f"""
    **Project:** Real-Time Public Transport Delay Prediction — Bengaluru

    **Domain:** Machine Learning | Time-Series Forecasting | Regression | Classification

    **Dataset:** BMTC GTFS Aggregated Data (4,655 real Bengaluru bus stops
    with trip counts, route counts, and GPS coordinates)

    **Team:** 2-Member Project · 300 Marks

    ---

    #### Model Architecture (Hybrid)

    | Model | Type | Scope | Purpose |
    |---|---|---|---|
    | **XGBoost** | ML Regression | All {n_stops:,} stops | Live delay prediction (primary) |
    | **Delay Classifier** | Rule-based + ML | All stops | 3-class delay severity label |
    | LSTM (Bi-directional) | Deep Learning | Busiest stop | Academic comparison |
    | ARIMA | Statistical | Busiest stop | Time-series baseline |
    | SARIMA | Statistical | Busiest stop | Seasonal time-series baseline |
    | Prophet | Time-series | Top {n_prophet} high-delay stops | 24-hr forecast chart |

    ---

    #### New Features (v2)
    - 🧠 **Delay Classifier** — classifies each corridor prediction into On-Time / Minor Delay / Major Delay with probability breakdown and feature impact
    - 🛠️ **Admin Dashboard** — network-level KPIs, worst/best stop heatmaps, delay distribution, hourly heatmap, rain impact analysis, batch stop audit
    - 🌦️ **Live weather** — OpenWeatherMap API, auto-sets rain toggle every 10 min
    - 🕐 **Current IST time default** — rounds to nearest 30-min slot
    - 🗺️ **GTFS bus lookup** — proper stop→trip→route join on BMTC GTFS data

    ---

    #### How Prediction Works
    1. User types FROM and TO stop — fuzzy search finds the closest match
    2. Travel date selected from calendar (past dates hidden)
    3. Time selected from 30-min slots — defaults to current IST time
    4. Rain status auto-detected from live weather API (overrideable)
    5. **XGBoost** predicts delay (minutes) using 19 engineered features
    6. **Delay Classifier** maps delay → severity class + confidence
    7. Bus numbers looked up from BMTC GTFS join (stops→trips→routes)
    8. If stop is in top {n_prophet} high-delay stops, **Prophet** adds 24-hr forecast

    ---

    #### Tools & Libraries
    Python · XGBoost · Prophet · Scikit-learn · TensorFlow/Keras ·
    Pandas · NumPy · Matplotlib · Streamlit · OpenWeatherMap API · Google Colab
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
    <hr>
    <p style='text-align:center;color:gray;font-size:0.8em'>
    BMTC Delay Prediction · Bengaluru · v2.0
    </p>
""", unsafe_allow_html=True)