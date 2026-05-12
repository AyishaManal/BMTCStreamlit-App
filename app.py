# ============================================================
# BMTC Bus Delay Predictor — Bengaluru  (Enhanced v3)
# Features: Login · Favourites · Travel History · ETA · Leaflet Map
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os
import re
import hashlib
import requests
import streamlit.components.v1 as components
from difflib import get_close_matches
from datetime import date, datetime, timedelta
import pytz
import warnings
warnings.filterwarnings("ignore")
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore


# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"


# ── OpenWeatherMap API key ────────────────────────────────────────────────────
# ▶ HOW TO SET: Streamlit Cloud → App Settings → Secrets → paste:
#
#   [openweather]
#   api_key = "YOUR_KEY_HERE"
#
# Get a free key at https://openweathermap.org/api
try:
    OWM_API_KEY = st.secrets["openweather"]["api_key"]
except (KeyError, FileNotFoundError):
    OWM_API_KEY = ""

# ══════════════════════════════════════════════════════════════════════════════
# FIREBASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

# Initialize Firebase Auth (client-side)
firebase_config = dict(st.secrets["firebase"])
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# initialize (Server-side)
if not firebase_admin._apps:
    
    firebase_admin_config = dict(st.secrets["firebase_admin"])

    # Fix newline issue explicitly
    firebase_admin_config["private_key"] = firebase_admin_config["private_key"].replace("\\n", "\n")

    cred = credentials.Certificate(firebase_admin_config)
    firebase_admin.initialize_app(cred)

db = firestore.client()
# ══════════════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def register_user(email: str, password: str, name: str):
    try:
        user = auth.create_user_with_email_and_password(email, password)

        db.collection("users").document(user["localId"]).set({
            "name": name,
            "email": email
        })

        return True, "Account created successfully!"

    except Exception as e:
        return False, str(e)   

def login_user(email: str, password: str):
    try:
        user = auth.sign_in_with_email_and_password(email, password)

        # Fetch user profile
        doc = db.collection("users").document(user["localId"]).get()
        data = doc.to_dict()

        return {
            "id": user["localId"],
            "email": email,
            "name": data.get("name", "User")
        }

    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# FAVORITES (Firestore)
# ══════════════════════════════════════════════════════════════════════════════

def add_favourite(user_id, label, from_stop, to_stop):
    try:
        _ts, _ref = db.collection("favorites").add({
            "user_id"  : user_id,
            "label"    : label,
            "from_stop": from_stop,
            "to_stop"  : to_stop,
            "added"    : firestore.SERVER_TIMESTAMP,
        })
        return True
    except Exception as e:
        st.session_state["fav_error"] = str(e)
        return False


def get_favourites(user_id):
    docs = db.collection("favorites").where(filter=firestore.FieldFilter("user_id", "==", user_id)).stream()
    return [doc.to_dict() | {"id": doc.id} for doc in docs]


def delete_favourite(fav_id):
    db.collection("favorites").document(fav_id).delete()

# ── Feedback helper ───────────────────────────────────────────────────────────
def save_feedback(user_id, user_name, rating, category, message):
    try:
        _ts, _ref = db.collection("feedback").add({
            "user_id"  : user_id,
            "user_name": user_name,
            "rating"   : rating,
            "category" : category,
            "message"  : message,
            "submitted": firestore.SERVER_TIMESTAMP,
        })
        return True
    except Exception as e:
        st.session_state["feedback_error"] = str(e)
        return False
    
# ══════════════════════════════════════════════════════════════════════════════
# TRAVEL HISTORY (Firestore)
# ══════════════════════════════════════════════════════════════════════════════

def save_history(user_id, from_stop, to_stop, travel_date, travel_time,
                 src_delay, dst_delay, is_rain):

    db.collection("travel_history").add({
        "user_id": user_id,
        "from_stop": from_stop,
        "to_stop": to_stop,
        "travel_date": str(travel_date),
        "travel_time": travel_time,
        "src_delay": src_delay,
        "dst_delay": dst_delay,
        "is_rain": is_rain,
        "searched": firestore.SERVER_TIMESTAMP
    })


def get_history(user_id, limit=30):
    import pytz
    docs = (
        db.collection("travel_history")
        .where(filter=firestore.FieldFilter("user_id", "==", user_id))
        .stream()
    )
    data = [doc.to_dict() for doc in docs]
    data = sorted(
        data,
        key=lambda x: x.get("searched") or datetime.min.replace(tzinfo=pytz.utc),
        reverse=True,
    )
    return data[:limit]

# ══════════════════════════════════════════════════════════════════════════════
# AUTH UI
# ══════════════════════════════════════════════════════════════════════════════

if "user" not in st.session_state:
    st.session_state["user"] = None


def show_auth():
    st.markdown("""
        <h1 style='color:#1A3A5C;margin-bottom:0'>🚌 BMTC Delay Predictor</h1>
        <p style='color:gray;margin-top:4px'>Bengaluru · ML-Powered Bus Delay Forecasting</p>
        <hr>
    """, unsafe_allow_html=True)

    tab_login, tab_reg = st.tabs(["🔑 Login", "📝 Register"])

    # ── LOGIN ────────────────────────────────────────────────────────────────
    with tab_login:
        st.subheader("Welcome back")

        email = st.text_input("Email", key="li_email")
        pw    = st.text_input("Password", type="password", key="li_pw")

        if st.button("Login", type="primary", use_container_width=True):
            if not email or not pw:
                st.error("Please fill in both fields.")
            else:
                user = login_user(email, pw)

                if user:
                    st.session_state["user"] = user
                    st.success(f"Welcome back, {user['name']}! 👋")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

    # ── REGISTER ─────────────────────────────────────────────────────────────
    with tab_reg:
        st.subheader("Create an account")

        name  = st.text_input("Full Name")
        email = st.text_input("Email")
        pw    = st.text_input("Password (min 6 chars)", type="password")
        pw2   = st.text_input("Confirm Password", type="password")

        if st.button("Create Account", type="primary", use_container_width=True):
            if not all([name, email, pw, pw2]):
                st.error("Please fill in all fields.")
            elif len(pw) < 6:
                st.error("Password must be at least 6 characters.")
            elif pw != pw2:
                st.error("Passwords do not match.")
            else:
                ok, msg = register_user(email, pw, name)
                if ok:
                    st.success(msg + " Please log in.")
                else:
                    st.error(msg)

# ══════════════════════════════════════════════════════════════════════════════
# LOGIN CHECK
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state["user"]:
    show_auth()
    st.stop()

# ── CURRENT USER ──────────────────────────────────────────────────────────────
CUR_USER    = st.session_state["user"]
CUR_USER_ID = CUR_USER["id"]


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=600)
def fetch_bengaluru_weather():
    if not OWM_API_KEY:
        return None, "no_key", None, None
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": 12.9716, "lon": 77.5946,
                    "appid": OWM_API_KEY, "units": "metric"},
            timeout=5,
        )
        data = r.json()
        wid  = data["weather"][0]["id"]
        desc = data["weather"][0]["description"].capitalize()
        temp = round(data["main"]["temp"], 1)
        icon = f"https://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png"
        return int(wid < 700), desc, temp, icon
    except Exception as e:
        return None, str(e), None, None

# ── IST time helpers ──────────────────────────────────────────────────────────
def get_ist_now():
    return datetime.now(pytz.timezone("Asia/Kolkata"))

def build_time_slots():
    labels, values = [], []
    for h in range(24):
        for m in (0, 30):
            labels.append(f"{h:02d}:{m:02d}")
            values.append((h, m))
    return labels, values

TIME_LABELS, TIME_VALUES = build_time_slots()

def slot_index_for(hour, minute):
    target  = hour * 60 + minute
    rounded = round(target / 30) * 30
    rounded = min(rounded, 23 * 60 + 30)
    h, m    = divmod(rounded, 60)
    try:
        return TIME_VALUES.index((h, m))
    except ValueError:
        return 0

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════
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
    safe = stop_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    path = os.path.join(MODEL_DIR, "prophet", f"{safe}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return model_from_json(f.read())

try:
    xgb_model, stop_summary, metadata = load_assets()
    final_results = load_results()
    prophet_stops = load_prophet_stops()
    FEATURES      = metadata["features"]
    all_stops     = sorted(stop_summary["stop_name"].tolist())
except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.info("Make sure models/ and outputs/ folders are present in your GitHub repo.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# GTFS BUS LOOKUP
# ══════════════════════════════════════════════════════════════════════════════
GTFS_DIR = os.path.join(MODEL_DIR, "gtfs")

@st.cache_resource(show_spinner=False)
def load_gtfs_tables():
    required = ["stops.txt", "stop_times.txt", "trips.txt", "routes.txt"]
    paths    = {f: os.path.join(GTFS_DIR, f) for f in required}
    if not all(os.path.exists(p) for p in paths.values()):
        return {}, False

    stops = pd.read_csv(paths["stops.txt"], usecols=["stop_id", "stop_name"], dtype=str).dropna()
    stops["stop_name_lower"] = stops["stop_name"].str.lower().str.strip()

    routes = pd.read_csv(paths["routes.txt"], usecols=["route_id", "route_short_name"], dtype=str).dropna()
    route_id_to_short = routes.set_index("route_id")["route_short_name"].to_dict()

    trips = pd.read_csv(paths["trips.txt"], usecols=["trip_id", "route_id"], dtype=str).dropna()
    trip_id_to_route = trips.set_index("trip_id")["route_id"].to_dict()

    stop_times = pd.read_csv(paths["stop_times.txt"], usecols=["trip_id", "stop_id"], dtype=str).dropna()
    stop_times["route_id"]         = stop_times["trip_id"].map(trip_id_to_route)
    stop_times["route_short_name"] = stop_times["route_id"].map(route_id_to_short)
    stop_times["stop_name_lower"]  = stop_times["stop_id"].map(
        stops.set_index("stop_id")["stop_name_lower"]
    )
    stop_times = stop_times.dropna(subset=["stop_name_lower", "route_short_name"])

    lookup: dict = {}
    for row in stop_times[["stop_name_lower", "route_short_name"]].itertuples(index=False):
        lookup.setdefault(row.stop_name_lower, set()).add(row.route_short_name)
    return lookup, True

@st.cache_resource(show_spinner=False)
def load_gtfs_coords():
    """Returns dict: stop_name_lower → (lat, lon)"""
    path = os.path.join(GTFS_DIR, "stops.txt")
    if not os.path.exists(path):
        return {}
    stops = pd.read_csv(path, usecols=["stop_name", "stop_lat", "stop_lon"], dtype=str).dropna()
    coords = {}
    for _, row in stops.iterrows():
        try:
            coords[row["stop_name"].lower().strip()] = (
                float(row["stop_lat"]), float(row["stop_lon"])
            )
        except Exception:
            pass
    return coords

@st.cache_resource(show_spinner=False)
def load_stop_timetable():
    """Returns dict: stop_name_lower → list of (route_short_name, arrival_time_str)
       arrival_time_str is HH:MM (24-hr), handles GTFS times > 23:59 (e.g. 25:30)."""
    required = ["stops.txt", "stop_times.txt", "trips.txt", "routes.txt"]
    paths    = {f: os.path.join(GTFS_DIR, f) for f in required}
    if not all(os.path.exists(p) for p in paths.values()):
        return {}

    stops  = pd.read_csv(paths["stops.txt"],
                         usecols=["stop_id", "stop_name"], dtype=str).dropna()
    trips  = pd.read_csv(paths["trips.txt"],
                         usecols=["trip_id", "route_id"], dtype=str).dropna()
    routes = pd.read_csv(paths["routes.txt"],
                         usecols=["route_id", "route_short_name"], dtype=str).dropna()
    stop_times = pd.read_csv(paths["stop_times.txt"],
                             usecols=["trip_id", "stop_id", "arrival_time"],
                             dtype=str).dropna()

    # Relational join
    merged = (stop_times
              .merge(trips,  on="trip_id")
              .merge(routes, on="route_id")
              .merge(stops,  on="stop_id"))

    merged["stop_name_lower"] = merged["stop_name"].str.lower().str.strip()

    timetable = {}
    for _, row in merged[["stop_name_lower", "route_short_name", "arrival_time"]].iterrows():
        key = row["stop_name_lower"]
        timetable.setdefault(key, []).append(
            (row["route_short_name"], row["arrival_time"])
        )
    return timetable


def _parse_gtfs_time(time_str):
    """Parse GTFS arrival_time (HH:MM:SS, may exceed 23:59) → total minutes from midnight."""
    try:
        parts = time_str.strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return -1


def _fmt_gtfs_time(time_str):
    """Format GTFS HH:MM:SS → display string HH:MM (wraps >23h back to next day)."""
    try:
        parts  = time_str.strip().split(":")
        h, m   = int(parts[0]), int(parts[1])
        h_disp = h % 24          # wrap GTFS times like 25:30 → 01:30
        return f"{h_disp:02d}:{m:02d}"
    except Exception:
        return time_str[:5]


def get_upcoming_buses(stop_name, selected_hour, selected_minute, window_min=90):
    """Return list of (route_short_name, display_time) arriving within window_min
       minutes of selected_hour:selected_minute at stop_name."""
    timetable = load_stop_timetable()
    if not timetable:
        return []

    key = _best_gtfs_match(stop_name.lower().strip(), timetable)
    if not key:
        return []

    selected_total = selected_hour * 60 + selected_minute
    upcoming = []

    for route, arr_time in timetable[key]:
        arr_total = _parse_gtfs_time(arr_time)
        if arr_total < 0:
            continue
        # Show buses arriving from now up to window_min minutes ahead
        if selected_total <= arr_total <= selected_total + window_min:
            upcoming.append((route, _fmt_gtfs_time(arr_time), arr_total))

    # Deduplicate: keep earliest arrival per route
    seen   = {}
    for route, disp, total in upcoming:
        if route not in seen or total < seen[route][1]:
            seen[route] = (disp, total)

    # Sort by arrival time
    result = sorted(
        [(route, disp) for route, (disp, _) in seen.items()],
        key=lambda x: x[1]
    )
    return result[:15]   # cap at 15 rows


def _best_gtfs_match(query, lookup):
    q = query.lower().strip()
    if q in lookup: return q
    substr = [k for k in lookup if q in k]
    if substr: return min(substr, key=len)
    substr2 = [k for k in lookup if k in q]
    if substr2: return max(substr2, key=len)
    matches = get_close_matches(q, list(lookup.keys()), n=1, cutoff=0.6)
    return matches[0] if matches else None

def find_buses(src, dst):
    lookup, gtfs_ok = load_gtfs_tables()
    if not gtfs_ok:
        return {"buses": [], "source": "none",
                "note": "GTFS files not found in models/gtfs/.", "hops": 0}
    src_key = _best_gtfs_match(src, lookup)
    dst_key = _best_gtfs_match(dst, lookup)
    if not src_key:
        return {"buses": [], "source": "none",
                "note": f"Stop not found in GTFS: '{src}'.", "hops": 0}
    if not dst_key:
        return {"buses": [], "source": "none",
                "note": f"Stop not found in GTFS: '{dst}'.", "hops": 0}
    direct = sorted(lookup[src_key] & lookup[dst_key])
    if direct:
        return {"buses": direct[:10], "source": "gtfs",
                "note": f"BMTC GTFS · {len(direct)} direct route(s)", "hops": 1}
    src_list = sorted(lookup[src_key])[:5]
    dst_list = sorted(lookup[dst_key])[:5]
    return {"buses": [], "source": "none",
            "note": (f"No direct bus. At {src}: {', '.join(src_list) or 'none'}. "
                     f"At {dst}: {', '.join(dst_list) or 'none'}."), "hops": 2}

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def find_stop(query, n=6):
    query = query.lower().strip()
    exact = [s for s in all_stops if query in s.lower()]
    if exact: return exact[:n]
    fuzzy = get_close_matches(query, [s.lower() for s in all_stops], n=n, cutoff=0.4)
    return [s for s in all_stops if s.lower() in fuzzy]

def build_features(stop_name, hour, dow, month, is_rain):
    row = stop_summary[stop_summary["stop_name"] == stop_name]
    if row.empty: return None, None
    s     = row.iloc[0]
    avg_d = float(s["avg_delay"])
    input_dict = {
        "factor"      : float(s["factor"]),
        "trip_count"  : float(s["trip_count"]),
        "route_count" : float(s["route_count"]),
        "hour"        : hour,
        "day_of_week" : dow,
        "month"       : month,
        "is_weekend"  : int(dow >= 5),
        "is_rush"     : int(hour in [7, 8, 9, 17, 18, 19]),
        "is_rain"     : is_rain,
        "hour_sin"    : np.sin(2 * np.pi * hour / 24),
        "hour_cos"    : np.cos(2 * np.pi * hour / 24),
        "dow_sin"     : np.sin(2 * np.pi * dow  / 7),
        "dow_cos"     : np.cos(2 * np.pi * dow  / 7),
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
    if X is None: return 0.0
    return round(float(np.clip(xgb_model.predict(X)[0], 0, None)), 1)

def get_status(delay):
    if delay < 3:  return "✅ On Time",      "#065F46", "#D1FAE5"
    if delay < 8:  return "⚠️ Minor Delay",  "#92400E", "#FEF3C7"
    return               "🔴 Major Delay",   "#991B1B", "#FEE2E2"

def compute_eta(hour, minute, delay_min):
    """Add predicted delay to departure time → ETA string."""
    base = datetime(2000, 1, 1, hour, minute)
    eta  = base + timedelta(minutes=delay_min)
    diff = int(delay_min)
    return eta.strftime("%H:%M"), diff

# ══════════════════════════════════════════════════════════════════════════════
# LEAFLET MAP COMPONENT
# ══════════════════════════════════════════════════════════════════════════════
def render_leaflet_map(src_stop: str, dst_stop: str,
                       src_delay: float, dst_delay: float) -> None:
    """
    Renders an interactive Leaflet map with:
    - Colour-coded markers for FROM / TO stops
    - A polyline connecting them (if coords available)
    - Delay info in popups

    ▶ REQUIRES: stops.txt in models/gtfs/ with stop_lat & stop_lon columns.
    ▶ Uses OpenStreetMap tiles — no API key needed.
    ▶ If you want Mapbox tiles instead, add your token below and swap the
      tileLayer URL to:
        https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{z}/{x}/{y}?access_token=YOUR_TOKEN
    """
    coords = load_gtfs_coords()

    def get_coord(stop_name):
        key = _best_gtfs_match(stop_name.lower(), coords) if coords else None
        if key and key in coords:
            return coords[key]
        # Fallback: Bengaluru city centre
        return (12.9716, 77.5946)

    src_lat, src_lon = get_coord(src_stop)
    dst_lat, dst_lon = get_coord(dst_stop)

    def delay_color(d):
        if d < 3:  return "#22c55e"   # green
        if d < 8:  return "#f59e0b"   # amber
        return            "#ef4444"   # red

    src_col = delay_color(src_delay)
    dst_col = delay_color(dst_delay)

    src_status = get_status(src_delay)[0]
    dst_status = get_status(dst_delay)[0]

    center_lat = (src_lat + dst_lat) / 2
    center_lon = (src_lon + dst_lon) / 2

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <link rel="stylesheet"
        href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ margin:0; padding:0; }}
    #map {{ width:100%; height:420px; border-radius:12px; }}
    .legend {{
      background:white; padding:10px 14px; border-radius:8px;
      box-shadow:0 1px 6px rgba(0,0,0,0.2); font-size:12px; line-height:1.8;
    }}
  </style>
</head>
<body>
<div id="map"></div>
<script>
  const map = L.map('map').setView([{center_lat}, {center_lon}], 13);

  // ── Tile layer (OpenStreetMap — free, no key needed) ──────────────────────
  // To use Mapbox: replace the URL with your Mapbox tile URL + access_token
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '© <a href="https://openstreetmap.org">OpenStreetMap</a>',
    maxZoom: 19
  }}).addTo(map);

  // ── Helper: coloured circle marker ───────────────────────────────────────
  function circleMarker(lat, lon, color, title, delay, status) {{
    const marker = L.circleMarker([lat, lon], {{
      radius: 14,
      fillColor: color,
      color: '#fff',
      weight: 3,
      opacity: 1,
      fillOpacity: 0.9
    }}).addTo(map);
    marker.bindPopup(`
      <b>${{title}}</b><br>
      🕐 Predicted delay: <b>${{delay}} min</b><br>
      ${{status}}
    `, {{ maxWidth: 180 }});
    return marker;
  }}

  const srcMarker = circleMarker(
    {src_lat}, {src_lon},
    '{src_col}', '📍 FROM: {src_stop}',
    {src_delay}, '{src_status}'
  );

  const dstMarker = circleMarker(
    {dst_lat}, {dst_lon},
    '{dst_col}', '🏁 TO: {dst_stop}',
    {dst_delay}, '{dst_status}'
  );

  // ── Polyline connecting the two stops ─────────────────────────────────────
  const routeLine = L.polyline(
    [[{src_lat}, {src_lon}], [{dst_lat}, {dst_lon}]],
    {{ color: '#1A3A5C', weight: 3, opacity: 0.7, dashArray: '8 6' }}
  ).addTo(map);

  // ── Open FROM popup by default ────────────────────────────────────────────
  srcMarker.openPopup();

  // ── Fit map to show both stops ────────────────────────────────────────────
  const bounds = L.latLngBounds(
    [{src_lat}, {src_lon}], [{dst_lat}, {dst_lon}]
  );
  map.fitBounds(bounds, {{ padding: [40, 40] }});

  // ── Legend ────────────────────────────────────────────────────────────────
  const legend = L.control({{ position: 'bottomright' }});
  legend.onAdd = function () {{
    const div = L.DomUtil.create('div', 'legend');
    div.innerHTML = `
      <b>Delay Severity</b><br>
      <span style="color:#22c55e">●</span> On-Time (&lt;3 min)<br>
      <span style="color:#f59e0b">●</span> Minor (3–8 min)<br>
      <span style="color:#ef4444">●</span> Major (&gt;8 min)
    `;
    return div;
  }};
  legend.addTo(map);
</script>
</body>
</html>
"""
    components.html(html, height=430, scrolling=False)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP HEADER + NAV
# ══════════════════════════════════════════════════════════════════════════════
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.markdown(f"""
        <h1 style='color:#1A3A5C;margin-bottom:0'>🚌 BMTC Delay Predictor</h1>
        <p style='color:gray;margin-top:4px'>
            Bengaluru · ML-Powered · Welcome, <b>{CUR_USER['name']}</b> 👋
        </p>
    """, unsafe_allow_html=True)
with col_h2:
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["user"] = None
        st.rerun()

st.markdown("<hr style='margin-top:0'>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Predict Journey",
    "📊 Model Results",
    "🧠 Delay Classifier",
    "ℹ️ About Project",
    "💬 Feedback",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — JOURNEY PREDICTOR  (with Favourites, History, ETA, Leaflet)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Sidebar-style Favourites panel ────────────────────────────────────────
    # Cache in session_state — cleared on save/delete to force a fresh Firestore read
    if "cached_favs" not in st.session_state:
        st.session_state["cached_favs"] = get_favourites(CUR_USER_ID)
    favs = st.session_state["cached_favs"]

    with st.expander("⭐ My Favourites", expanded=False):
        if favs:
            for fav in favs:
                fc1, fc2, fc3 = st.columns([3, 2, 1])
                with fc1:
                    st.markdown(f"**{fav['label']}**")
                    st.caption(f"📍 {fav['from_stop']} → 🏁 {fav['to_stop']}")
                with fc2:
                    if st.button("Use this route", key=f"use_fav_{fav['id']}"):
                        st.session_state["prefill_from"] = fav["from_stop"]
                        st.session_state["prefill_to"]   = fav["to_stop"]
                        st.session_state.pop("cached_favs", None)
                        st.rerun()
                with fc3:
                    if st.button("🗑️", key=f"del_fav_{fav['id']}"):
                        delete_favourite(fav["id"])
                        st.session_state.pop("cached_favs", None)
                        st.rerun()
                st.markdown("---")
        else:
            st.info("No favourites yet. Search a route and save it!")

    # ── Travel History panel ──────────────────────────────────────────────────
    with st.expander("🕓 Travel History", expanded=False):
        history = get_history(CUR_USER_ID)
        if history:
            hist_data = []
            for h in history:
                ts = h.get("searched")
                searched_str = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else (str(ts)[:16] if ts else "—")
                hist_data.append({
                    "Date"      : h.get("travel_date", "—"),
                    "Time"      : h.get("travel_time", "—"),
                    "From"      : h.get("from_stop", "—"),
                    "To"        : h.get("to_stop", "—"),
                    "From Delay": f"{h['src_delay']} min" if h.get("src_delay") is not None else "—",
                    "To Delay"  : f"{h['dst_delay']} min" if h.get("dst_delay") is not None else "—",
                    "Rain"      : "🌧️" if h.get("is_rain") else "☀️",
                    "Searched"  : searched_str,
                })
            st.dataframe(pd.DataFrame(hist_data), use_container_width=True, height=240)
        else:
            st.info("No history yet. Your searches will appear here.")

    st.subheader("Enter Your Journey Details")

    # ── Pre-fill from favourites ───────────────────────────────────────────────
    prefill_from = st.session_state.pop("prefill_from", "")
    prefill_to   = st.session_state.pop("prefill_to", "")

    col1, col2 = st.columns(2)
    with col1:
        from_input = st.text_input("📍 FROM Stop",
                                    value=prefill_from,
                                    placeholder="e.g. Halasuru, Hebbal")
    with col2:
        to_input   = st.text_input("🏁 TO Stop",
                                    value=prefill_to,
                                    placeholder="e.g. Majestic, Silk Board")

    # ── Date picker ────────────────────────────────────────────────────────────
    st.markdown("📅 **Select Travel Date**")
    today       = date.today()
    travel_date = st.date_input(" ", value=today,
                                 min_value=today, max_value=today + timedelta(days=30),
                                 label_visibility="collapsed")
    dow   = travel_date.weekday()
    month = travel_date.month
    day   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][dow]
    st.caption(f"📆 {travel_date.strftime('%d %B %Y')}  ({day})")

    # ── Time selector ─────────────────────────────────────────────────────────
    st.markdown("⏰ **Time of Travel**")
    now_ist     = get_ist_now()
    default_idx = slot_index_for(now_ist.hour, now_ist.minute)
    st.caption(f"🕐 Current Bengaluru time: **{now_ist.strftime('%H:%M IST')}**")
    selected_label = st.selectbox(" ", options=TIME_LABELS, index=default_idx,
                                   label_visibility="collapsed")
    hour, minute = TIME_VALUES[TIME_LABELS.index(selected_label)]

    if   hour in [7, 8, 9]:      st.caption("🔴 AM Rush Hour — expect higher delays")
    elif hour in [17, 18, 19]:   st.caption("🔴 PM Rush Hour — expect higher delays")
    elif 0 <= hour <= 5:         st.caption("🌙 Late Night — minimal traffic expected")
    else:                        st.caption("🟡 Normal Hours")

    # ── Live weather ──────────────────────────────────────────────────────────
    st.markdown("🌦️ **Weather Conditions**")
    live_rain, live_desc, live_temp, live_icon = fetch_bengaluru_weather()

    if live_rain is not None:
        wc1, wc2 = st.columns([1, 4])
        with wc1:
            if live_icon: st.image(live_icon, width=60)
        with wc2:
            st.markdown(f"**Live weather:** {'🌧️' if live_rain else '☀️'} {live_desc}  |  🌡️ {live_temp}°C")
            st.caption("Auto-detected via OpenWeatherMap · updates every 10 min")
        is_rain = int(st.toggle("🌧️ Raining? (override if needed)", value=bool(live_rain)))
    elif live_desc == "no_key":
        st.info("💡 Add your OpenWeatherMap API key in Streamlit secrets for live rain detection.")
        is_rain = int(st.toggle("🌧️ Raining?", value=False))
    else:
        st.warning(f"Weather fetch failed ({live_desc}). Using manual toggle.")
        is_rain = int(st.toggle("🌧️ Raining?", value=False))

    if month in [6, 7, 8, 9]:
        st.caption("☔ Monsoon season — rain likely to add 2–5 min extra delay")

    # ── Stop fuzzy search ──────────────────────────────────────────────────────
    src_stop, dst_stop = None, None
    if from_input:
        matches = find_stop(from_input)
        if matches:   src_stop = st.selectbox("Select FROM stop:", matches, key="src")
        else:          st.warning("No FROM stop found. Try a different name.")
    if to_input:
        matches = find_stop(to_input)
        if matches:   dst_stop = st.selectbox("Select TO stop:", matches, key="dst")
        else:          st.warning("No TO stop found. Try a different name.")

    # ── Predict button ────────────────────────────────────────────────────────
    if st.button("🔍 Predict Delay", type="primary", use_container_width=True):
        if not src_stop or not dst_stop:
            st.error("Please enter both FROM and TO stops.")
        elif src_stop == dst_stop:
            st.error("FROM and TO stops cannot be the same.")
        else:
            src_delay = predict_delay(src_stop, hour, dow, month, is_rain)
            dst_delay = predict_delay(dst_stop, hour, dow, month, is_rain)

            src_label, src_color, src_bg = get_status(src_delay)
            dst_label, dst_color, dst_bg = get_status(dst_delay)
            worse = max(src_delay, dst_delay)

            with st.spinner("🔍 Looking up bus numbers..."):
                bus_result = find_buses(src_stop, dst_stop)

            buses  = bus_result["buses"]
            source = bus_result["source"]
            note   = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1 (\2)", bus_result["note"])

            # ── ETA ───────────────────────────────────────────────────────────
            src_eta, src_extra = compute_eta(hour, minute, src_delay)
            dst_eta, dst_extra = compute_eta(hour, minute, dst_delay)

            # ── Travel tip ────────────────────────────────────────────────────
            weather_icon = "🌧️ Rain" if is_rain else "☀️ Clear"
            rush_note    = " · 🔴 AM Rush" if hour in [7,8,9] else (
                           " · 🔴 PM Rush" if hour in [17,18,19] else "")
            if worse >= 8:
                tip_bg, tip_icon, tip_text = "#FEE2E2", "🔴", "Leave early — heavy delays expected!"
            elif worse >= 3:
                tip_bg, tip_icon, tip_text = "#FEF3C7", "⚠️", "Keep a 10-minute buffer for this journey."
            else:
                tip_bg, tip_icon, tip_text = "#D1FAE5", "✅", "Good time to travel — minimal delays expected."

            # ── Bus pills ─────────────────────────────────────────────────────
            PILL = ("display:inline-block;background:#1E3A5F;color:#E0F2FE;"
                    "border-radius:6px;padding:4px 10px;margin:3px 4px 3px 0;"
                    "font-weight:700;font-size:0.9rem;letter-spacing:0.02em")
            dot_color, dot_label = {
                "gtfs"  : ("#16A34A", "🟢 BMTC GTFS data"),
                "none"  : ("#9CA3AF", "⚪ Not found"),
            }.get(source, ("#9CA3AF", ""))

            if buses:
                pills    = "".join(f'<span style="{PILL}">{b}</span>' for b in buses)
                bus_html = (
                    '<div style="margin-top:14px;padding-top:14px;border-top:1px solid #E2E8F0">'
                    '<p style="margin:0 0 8px 0;font-weight:700;color:#1A3A5C;font-size:0.9rem">'
                    f'🚌 Buses on this corridor</p><div style="line-height:2">{pills}</div>'
                    f'<p style="margin:8px 0 0 0;font-size:0.75rem;color:{dot_color}">'
                    f'{dot_label} · {note}</p></div>'
                )
            else:
                bus_html = (
                    '<div style="margin-top:14px;padding-top:14px;border-top:1px solid #E2E8F0">'
                    f'<p style="margin:0;font-size:0.85rem;color:#6B7280">'
                    f'🚌 Bus numbers unavailable.<br>'
                    f'<span style="font-size:0.75rem">{note}</span></p></div>'
                )

            # ── Result card ───────────────────────────────────────────────────
            st.markdown("---")
            card_html = f"""
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  body{{margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:transparent}}
  .card{{background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:22px 24px 18px;
          box-shadow:0 2px 12px rgba(0,0,0,.07)}}
  .meta{{font-size:.8rem;color:#64748B;margin:4px 0 16px}}
  .rh{{display:flex;align-items:center;gap:8px;margin-bottom:4px}}
  .rh span{{font-size:1.05rem;font-weight:700;color:#1A3A5C}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
  .dc{{border-radius:10px;padding:16px 18px}}
  .dc .role{{margin:0 0 2px;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em}}
  .dc .sn{{margin:0 0 6px;font-size:.85rem;font-weight:600}}
  .dc .dv{{margin:0;font-size:2rem;font-weight:800;line-height:1}}
  .dc .dv span{{font-size:1rem;font-weight:500}}
  .dc .st{{margin:2px 0 0;font-size:.85rem}}
  .dc .eta{{margin:4px 0 0;font-size:.78rem;opacity:.8}}
  .tip{{margin-top:14px;border-radius:8px;padding:10px 14px;font-size:.88rem;font-weight:600;color:#1A3A5C}}
</style></head><body>
<div class="card">
  <div class="rh">
    <span>📍 {src_stop}</span>
    <span style="color:#64748B;font-size:1.1rem">→</span>
    <span>🏁 {dst_stop}</span>
  </div>
  <p class="meta">📅 {travel_date.strftime('%d %b %Y')} ({day}) &nbsp;·&nbsp;
     ⏰ {selected_label} &nbsp;·&nbsp; {weather_icon}{rush_note}</p>
  <div class="grid">
    <div class="dc" style="background:{src_bg};border-left:5px solid {src_color}">
      <p class="role" style="color:{src_color}">FROM</p>
      <p class="sn"   style="color:{src_color}">{src_stop}</p>
      <p class="dv"   style="color:{src_color}">{src_delay}<span> min</span></p>
      <p class="st"   style="color:{src_color}">{src_label}</p>
      <p class="eta"  style="color:{src_color}">🕐 ETA: {src_eta} (+{src_extra} min)</p>
    </div>
    <div class="dc" style="background:{dst_bg};border-left:5px solid {dst_color}">
      <p class="role" style="color:{dst_color}">TO</p>
      <p class="sn"   style="color:{dst_color}">{dst_stop}</p>
      <p class="dv"   style="color:{dst_color}">{dst_delay}<span> min</span></p>
      <p class="st"   style="color:{dst_color}">{dst_label}</p>
      <p class="eta"  style="color:{dst_color}">🕐 ETA: {dst_eta} (+{dst_extra} min)</p>
    </div>
  </div>
  {bus_html}
  <div class="tip" style="background:{tip_bg}">{tip_icon} {tip_text}</div>
</div></body></html>"""
            components.html(card_html, height=450, scrolling=False)

            # ── Upcoming Bus Timetable ────────────────────────────────────────
            st.markdown("#### 🕐 Upcoming Buses at Your Stop")
            st.caption(
                f"Buses arriving at **{src_stop}** within 90 minutes of "
                f"**{selected_label}** (scheduled times from BMTC GTFS)"
            )
            with st.spinner("Loading timetable..."):
                upcoming_buses = get_upcoming_buses(src_stop, hour, minute, window_min=90)

            if upcoming_buses:
                # Build colour-coded HTML table
                rows_html = ""
                for i, (route, arr_time) in enumerate(upcoming_buses):
                    # Colour row by how soon the bus arrives
                    try:
                        arr_h, arr_m = map(int, arr_time.split(":"))
                        arr_total    = arr_h * 60 + arr_m
                        sel_total    = hour * 60 + minute
                        diff_min     = arr_total - sel_total
                    except Exception:
                        diff_min = 99

                    if diff_min <= 15:
                        row_bg   = "#FEF3C7"
                        time_col = "#92400E"
                        badge    = f'<span style="font-size:.7rem;background:#F59E0B;color:#fff;padding:1px 6px;border-radius:4px;margin-left:6px">in {diff_min} min</span>'
                    elif diff_min <= 30:
                        row_bg   = "#F0FDF4"
                        time_col = "#065F46"
                        badge    = f'<span style="font-size:.7rem;background:#22C55E;color:#fff;padding:1px 6px;border-radius:4px;margin-left:6px">in {diff_min} min</span>'
                    else:
                        row_bg   = "#F8FAFC"
                        time_col = "#1E3A5F"
                        badge    = f'<span style="font-size:.7rem;color:#94A3B8;margin-left:6px">in {diff_min} min</span>'

                    rows_html += f"""
                    <tr style="background:{row_bg}">
                      <td style="padding:8px 14px;font-weight:700;color:#1A3A5C;font-size:.9rem">{route}</td>
                      <td style="padding:8px 14px;font-weight:800;color:{time_col};font-size:.95rem">
                        {arr_time}{badge}
                      </td>
                    </tr>"""

                timetable_html = f"""
<html><head><meta charset="utf-8">
<style>
  body{{margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}
  table{{width:100%;border-collapse:collapse;border-radius:10px;overflow:hidden;
         border:1px solid #E2E8F0;box-shadow:0 1px 6px rgba(0,0,0,.06)}}
  thead tr{{background:#1A3A5C}}
  thead td{{padding:9px 14px;color:#fff;font-weight:700;font-size:.8rem;
             text-transform:uppercase;letter-spacing:.05em}}
  tbody tr:hover{{filter:brightness(0.97)}}
</style></head><body>
<table>
  <thead><tr>
    <td>Bus Number</td>
    <td>Arrives At · {src_stop[:30]}</td>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
<p style="margin:6px 0 0;font-size:.72rem;color:#94A3B8">
  🟡 within 15 min &nbsp;·&nbsp; 🟢 within 30 min &nbsp;·&nbsp;
  Scheduled times only — actual arrival may vary by predicted delay above.
</p>
</body></html>"""
                components.html(timetable_html,
                                height=min(80 + len(upcoming_buses) * 42, 560),
                                scrolling=True)
            else:
                st.info(
                    f"No scheduled buses found at **{src_stop}** between "
                    f"**{selected_label}** and "
                    f"**{_fmt_gtfs_time(f'{(hour*60+minute+90)//60:02d}:{(hour*60+minute+90)%60:02d}:00')}**. "
                    "This may mean the GTFS file does not cover this stop or "
                    "no buses are scheduled in this window."
                )

            # ── Save to history ────────────────────────────────────────────────
            save_history(CUR_USER_ID, src_stop, dst_stop, travel_date,
                         selected_label, src_delay, dst_delay, is_rain)

            # Store stops in session_state — Save button lives outside this block
            st.session_state["last_src"] = src_stop
            st.session_state["last_dst"] = dst_stop

            # ── Leaflet Map ────────────────────────────────────────────────────
            st.markdown("#### 🗺️ Route Map")
            st.caption("Colour-coded by predicted delay severity. Click markers for details.")
            render_leaflet_map(src_stop, dst_stop, src_delay, dst_delay)

            # ── 24-hr Forecast Chart ──────────────────────────────────────────
            st.markdown("#### 24-Hour Delay Forecast")
            src_has_prophet = src_stop in prophet_stops
            dst_has_prophet = dst_stop in prophet_stops
            hours = list(range(24))

            def get_24h(stop_name, has_prophet):
                if has_prophet:
                    m = load_prophet_model(stop_name)
                    if m:
                        future = pd.DataFrame({
                            "ds"     : pd.date_range("2024-07-01", periods=24, freq="h"),
                            "is_rush": [1 if h in [7,8,9,17,18,19] else 0 for h in range(24)]
                        })
                        fc = m.predict(future)
                        return (fc["yhat"].clip(lower=0).tolist(),
                                fc["yhat_lower"].clip(lower=0).tolist(),
                                fc["yhat_upper"].clip(lower=0).tolist())
                vals = [predict_delay(stop_name, h, dow, month, is_rain) for h in hours]
                return vals, None, None

            src_24, src_lo, src_hi = get_24h(src_stop, src_has_prophet)
            dst_24, dst_lo, dst_hi = get_24h(dst_stop, dst_has_prophet)

            fig, ax = plt.subplots(figsize=(10, 4))
            lbl_s = f"{src_stop[:22]} ({'Prophet' if src_has_prophet else 'XGBoost'})"
            lbl_d = f"{dst_stop[:22]} ({'Prophet' if dst_has_prophet else 'XGBoost'})"
            ax.plot(hours, src_24, label=lbl_s, color="steelblue", lw=2, marker="o", markersize=3)
            if src_lo: ax.fill_between(hours, src_lo, src_hi, alpha=0.12, color="steelblue")
            ax.plot(hours, dst_24, label=lbl_d, color="crimson",  lw=2, marker="s", markersize=3)
            if dst_lo: ax.fill_between(hours, dst_lo, dst_hi, alpha=0.12, color="crimson")

            ax.axvline(hour + minute/60, color="gray", ls="--", lw=1.5,
                       label=f"Your time ({selected_label})")
            ax.axvspan(7,  9,  alpha=0.10, color="red",    label="AM Rush")
            ax.axvspan(17, 19, alpha=0.10, color="orange", label="PM Rush")
            ax.set_xlabel("Hour of Day"); ax.set_ylabel("Predicted Delay (min)")
            ax.set_title(f"{travel_date.strftime('%d %b %Y')} ({day})"
                         f"{'  🌧️ Rain' if is_rain else ''}")
            ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xticks(range(0, 24, 2))
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    # ── Save Favourite UI — outside predict block so button click is always caught
    if st.session_state.get("last_src") and st.session_state.get("last_dst"):
        _src = st.session_state["last_src"]
        _dst = st.session_state["last_dst"]
        st.markdown("---")
        st.markdown("#### ⭐ Save This Route to Favourites")
        _c1, _c2 = st.columns([3, 1])
        with _c1:
            fav_label = st.text_input(
                "Give this route a label (optional)",
                placeholder=f"{_src} → {_dst}",
                key="fav_label_input",
            )
        with _c2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⭐ Save Favourite", key="save_fav_btn", use_container_width=True):
                resolved_label = fav_label.strip() or f"{_src} → {_dst}"
                ok = add_favourite(CUR_USER_ID, resolved_label, _src, _dst)
                if ok:
                    st.session_state.pop("cached_favs", None)  # force fresh read
                    st.toast(f"⭐ '{resolved_label}' saved!", icon="✅")
                    st.rerun()
                else:
                    err = st.session_state.pop("fav_error", "Unknown Firestore error")
                    st.error(f"❌ {err}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance Comparison")
    n_stops = metadata.get("n_stops", "~1,955")
    st.markdown(
        f"Models trained on 180 days of hourly delay data across "
        f"**{n_stops:,} BMTC stops**. "
        "Evaluated on a time-based held-out 20% test set."
    )
    st.dataframe(final_results.set_index("Model"), use_container_width=True)
    st.caption("RMSE and MAE in minutes — lower is better.")

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
        ax.set_ylabel("Error (minutes)"); ax.set_title("RMSE & MAE — All Models")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    for title, fname in [
        ("XGBoost Feature Importance",           "feature_importance.png"),
        ("XGBoost Training Curve",               "xgb_training_curve.png"),
        ("XGBoost — Actual vs Predicted",        "xgb_predictions.png"),
        ("Prophet 24-Hour Forecasts (Top 6)",    "prophet_forecasts.png"),
        ("ARIMA vs SARIMA — Actual vs Predicted","arima_sarima.png"),
        ("LSTM — Training Curves",               "lstm_training.png"),
        ("LSTM — Actual vs Predicted",           "lstm_predictions.png"),
        ("EDA — Delay Patterns",                 "eda.png"),
        ("Rain Effect on Delays",                "rain_effect.png"),
    ]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            st.markdown(f"#### {title}")
            st.image(path, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DELAY CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🧠 Delay Severity Classifier")
    st.markdown(
        "Classifies a stop's delay into **On-Time / Minor Delay / Major Delay** "
        "using the XGBoost regression output + domain thresholds."
    )
    st.markdown("---")

    # ── Stop Name ─────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        cls_stop_input = st.text_input("🔍 Stop Name", placeholder="e.g. Hebbal, Silk Board", key="cls_stop")
    with c2:
        cls_rain = int(st.toggle("🌧️ Rain", key="cls_rain"))

    # ── Date picker ───────────────────────────────────────────────────────────
    st.markdown("📅 **Select Travel Date**")
    cls_today       = date.today()
    cls_travel_date = st.date_input(
        " ",
        value=cls_today,
        min_value=cls_today,
        max_value=cls_today + timedelta(days=30),
        label_visibility="collapsed",
        key="cls_date",
    )
    cls_dow_int = cls_travel_date.weekday()
    cls_month   = cls_travel_date.month
    cls_day_name = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][cls_dow_int]
    cls_dow      = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][cls_dow_int]
    st.caption(f"📆 {cls_travel_date.strftime('%d %B %Y')}  ({cls_day_name})")

    # ── Time selector ─────────────────────────────────────────────────────────
    st.markdown("⏰ **Time of Travel**")
    cls_now_ist     = get_ist_now()
    cls_default_idx = slot_index_for(cls_now_ist.hour, cls_now_ist.minute)
    st.caption(f"🕐 Current Bengaluru time: **{cls_now_ist.strftime('%H:%M IST')}**")
    cls_selected_label = st.selectbox(
        " ",
        options=TIME_LABELS,
        index=cls_default_idx,
        label_visibility="collapsed",
        key="cls_time",
    )
    cls_hour, cls_minute = TIME_VALUES[TIME_LABELS.index(cls_selected_label)]

    if   cls_hour in [7, 8, 9]:    st.caption("🔴 AM Rush Hour — expect higher delays")
    elif cls_hour in [17, 18, 19]: st.caption("🔴 PM Rush Hour — expect higher delays")
    elif 0 <= cls_hour <= 5:       st.caption("🌙 Late Night — minimal traffic expected")
    else:                          st.caption("🟡 Normal Hours")

    # ── Stop fuzzy match ──────────────────────────────────────────────────────
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

                if raw_delay < 3:
                    cls_label = "✅ On-Time";     cls_color = "#065F46"; cls_bg = "#D1FAE5"
                    p_ontime  = max(0.0, min(1.0, 1.0 - raw_delay / 3.0))
                    p_minor   = 1.0 - p_ontime;  p_major = 0.0
                elif raw_delay < 8:
                    cls_label = "⚠️ Minor Delay"; cls_color = "#92400E"; cls_bg = "#FEF3C7"
                    t        = (raw_delay - 3) / 5.0
                    p_minor  = max(0.4, 1.0 - abs(t - 0.5))
                    p_ontime = max(0.0, 0.5 - t * 0.5)
                    p_major  = max(0.0, t * 0.5)
                    total    = p_ontime + p_minor + p_major
                    p_ontime /= total; p_minor /= total; p_major /= total
                else:
                    cls_label = "🔴 Major Delay"; cls_color = "#991B1B"; cls_bg = "#FEE2E2"
                    p_major  = min(1.0, 0.5 + (raw_delay - 8) / 20.0)
                    p_minor  = 1.0 - p_major;    p_ontime = 0.0

                feat_vals = X_cls.iloc[0].to_dict()
                try:
                    fi        = dict(zip(FEATURES, xgb_model.feature_importances_))
                    impact    = {f: abs(feat_vals.get(f, 0)) * fi.get(f, 0) for f in FEATURES}
                    top_feats = sorted(impact.items(), key=lambda x: x[1], reverse=True)[:6]
                except Exception:
                    top_feats = []

                prob_bar = lambda p, col, lbl: (
                    f'<div style="margin:6px 0">'
                    f'<div style="display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:2px">'
                    f'<span>{lbl}</span><span>{p*100:.1f}%</span></div>'
                    f'<div style="background:#E5E7EB;border-radius:4px;height:10px">'
                    f'<div style="background:{col};width:{p*100:.1f}%;height:10px;border-radius:4px"></div>'
                    f'</div></div>'
                )

                feat_colors = {
                    "is_rain"    : "#3B82F6", "is_rush"    : "#EF4444",
                    "hour"       : "#F59E0B", "factor"     : "#8B5CF6",
                    "trip_count" : "#06B6D4", "route_count": "#10B981",
                }
                feat_rows = ""
                for fn, fv in top_feats:
                    disp  = fn.replace("_", " ").title()
                    raw_v = feat_vals.get(fn, 0)
                    fc    = feat_colors.get(fn, "#6B7280")
                    feat_rows += (
                        f'<tr><td style="padding:4px 8px;font-size:.8rem">{disp}</td>'
                        f'<td style="padding:4px 8px;font-size:.8rem;text-align:right">'
                        f'<span style="background:{fc}22;color:{fc};padding:2px 6px;'
                        f'border-radius:4px;font-weight:600">{raw_v:.3f}</span></td></tr>'
                    )

                rush_tag = (
                    "🔴 AM Rush" if cls_hour in [7, 8, 9] else
                    "🔴 PM Rush" if cls_hour in [17, 18, 19] else ""
                )
                rain_tag = "🌧️ Rain" if cls_rain else "☀️ Clear"

                html = f"""<html><head><meta charset="utf-8">
<style>
  body{{margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}
  .wrap{{background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:22px 24px;box-shadow:0 2px 12px rgba(0,0,0,.07)}}
  .badge{{display:inline-block;padding:6px 16px;border-radius:8px;font-weight:800;font-size:1.1rem;margin-bottom:12px}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px}}
  .box{{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:14px}}
  .box h4{{margin:0 0 10px;font-size:.85rem;color:#64748B;font-weight:700;text-transform:uppercase;letter-spacing:.06em}}
  table{{width:100%;border-collapse:collapse}}
</style></head><body>
<div class="wrap">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
    <div class="badge" style="background:{cls_bg};color:{cls_color}">{cls_label}</div>
    <span style="font-size:.85rem;color:#64748B">
      {cls_stop} &nbsp;·&nbsp; {cls_selected_label} &nbsp;·&nbsp; {cls_day_name}
      &nbsp;·&nbsp; {cls_travel_date.strftime('%d %b %Y')}
      &nbsp;·&nbsp; {rain_tag}
      {"&nbsp;·&nbsp;" + rush_tag if rush_tag else ""}
    </span>
  </div>
  <p style="font-size:1.6rem;font-weight:800;color:{cls_color};margin:8px 0 0">
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
</div></body></html>"""
                components.html(html, height=310, scrolling=False)

                st.markdown("#### Compare Multiple Stops at This Time")
                sample_stops = (
                    stop_summary.nlargest(10, "avg_delay")["stop_name"].tolist() +
                    stop_summary.nsmallest(10, "avg_delay")["stop_name"].tolist()
                )
                batch_rows = []
                for sn in sample_stops:
                    d   = predict_delay(sn, cls_hour, cls_dow_int, cls_month, cls_rain)
                    lbl = "On-Time" if d < 3 else ("Minor Delay" if d < 8 else "Major Delay")
                    batch_rows.append({"Stop": sn, "Predicted Delay (min)": d, "Class": lbl})
                bdf = pd.DataFrame(batch_rows).sort_values("Predicted Delay (min)", ascending=False)

                fig_b, ax_b = plt.subplots(figsize=(10, 5))
                colors_b = [
                    "#EF4444" if c == "Major Delay" else
                    "#F59E0B" if c == "Minor Delay" else "#10B981"
                    for c in bdf["Class"]
                ]
                ax_b.barh(bdf["Stop"].str[:30], bdf["Predicted Delay (min)"],
                          color=colors_b, edgecolor="white")
                ax_b.axvline(3, color="#F59E0B", ls="--", lw=1.2, label="Minor (3 min)")
                ax_b.axvline(8, color="#EF4444", ls="--", lw=1.2, label="Major (8 min)")
                ax_b.set_xlabel("Predicted Delay (min)")
                ax_b.set_title(
                    f"Delay Classification — {cls_selected_label} · {cls_day_name} · "
                    f"{cls_travel_date.strftime('%d %b %Y')} · "
                    f"{'Rain' if cls_rain else 'Clear'}"
                )
                ax_b.legend(fontsize=8)
                ax_b.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_b)
                plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("About This Project")
    n_stops   = metadata.get("n_stops", "~1,955")
    n_prophet = len(prophet_stops)
    st.markdown(f"""
**Project:** Real-Time Public Transport Delay Prediction — Bengaluru

**Domain:** Machine Learning | Time-Series Forecasting | Regression | Classification

**Dataset:** BMTC GTFS Aggregated Data (4,655 real Bengaluru bus stops
with trip counts, route counts, and GPS coordinates)


---

#### Model Architecture (Hybrid)

| Model | Type | Scope | Purpose |
|---|---|---|---|
| **XGBoost** | ML Regression | All {n_stops:,} stops | Live delay prediction (primary) |
| **Delay Classifier** | Rule-based + ML | All stops | 3-class delay severity label |
| LSTM (Bi-directional) | Deep Learning | Busiest stop | Academic comparison |
| ARIMA | Statistical | Busiest stop | Time-series baseline |
| SARIMA | Statistical | Busiest stop | Seasonal baseline |
| Prophet | Time-series | Top {n_prophet} high-delay stops | 24-hr forecast chart |

---

#### New Features (v3)
- 🔐 **Login & Registration** — SQLite-backed user accounts (email + hashed password)
- ⭐ **Travel Favourites** — Save, name, and reuse frequent routes (persisted in DB)
- 🕓 **Travel History** — Every search saved automatically, viewable per user
- 🕐 **ETA Prediction** — Departure time + predicted delay = estimated arrival time
- 🗺️ **Leaflet Map** — Interactive route map with colour-coded delay markers
- 🌦️ **Live weather** — OpenWeatherMap API auto-sets rain toggle every 10 min
- 🗺️ **GTFS bus lookup** — Stop→trip→route join on BMTC GTFS data

---

#### How Prediction Works
1. User logs in → searches FROM and TO stop
2. XGBoost predicts delay (minutes) using 19 engineered features
3. Delay Classifier maps delay → severity class + confidence
4. ETA = departure time + predicted delay
5. Bus numbers looked up from BMTC GTFS join
6. Leaflet map shows colour-coded markers at both stops
7. If stop is in top {n_prophet} high-delay stops, Prophet adds 24-hr forecast
8. Search saved to user's travel history automatically

---

#### Tools & Libraries
Python · XGBoost · Prophet · Scikit-learn · TensorFlow/Keras ·
Pandas · NumPy · Matplotlib · Streamlit · SQLite · Leaflet.js ·
OpenWeatherMap API · Google Colab
""")
# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("💬 Share Your Feedback")
    st.markdown(
        "Help us improve the BMTC Delay Predictor. "
        "Your feedback is saved and reviewed by the team."
    )
    st.markdown("---")

    # ── Star rating ───────────────────────────────────────────────────────────
    st.markdown("#### ⭐ Rate Your Experience")
    rating = st.select_slider(
        " ",
        options=["⭐ 1 — Poor", "⭐⭐ 2 — Fair", "⭐⭐⭐ 3 — Good",
                 "⭐⭐⭐⭐ 4 — Very Good", "⭐⭐⭐⭐⭐ 5 — Excellent"],
        value="⭐⭐⭐ 3 — Good",
        label_visibility="collapsed",
        key="fb_rating",
    )

    # ── Category ──────────────────────────────────────────────────────────────
    st.markdown("#### 📂 Feedback Category")
    category = st.selectbox(
        " ",
        options=[
            "Prediction Accuracy",
            "App Speed & Performance",
            "Map & Visualisation",
            "Login & Account Features",
            "Favourites & Travel History",
            "Bus Number Lookup",
            "UI / User Experience",
            "Feature Request",
            "Bug Report",
            "Other",
        ],
        label_visibility="collapsed",
        key="fb_category",
    )

    # ── Message ───────────────────────────────────────────────────────────────
    st.markdown("#### 📝 Your Message")
    message = st.text_area(
        " ",
        placeholder="Tell us what you think — what worked well, what could be better, or any feature you'd like to see...",
        height=150,
        label_visibility="collapsed",
        key="fb_message",
    )

    # ── Submit ────────────────────────────────────────────────────────────────
    if st.button("📨 Submit Feedback", type="primary",
                 use_container_width=True, key="fb_submit"):
        if not message.strip():
            st.warning("⚠️ Please write a message before submitting.")
        else:
            ok = save_feedback(
                CUR_USER_ID,
                CUR_USER["name"],
                rating,
                category,
                message.strip(),
            )
            if ok:
                st.success(
                    f"✅ Thank you, {CUR_USER['name']}! "
                    "Your feedback has been submitted successfully."
                )
                st.balloons()
            else:
                err = st.session_state.pop("feedback_error", "Unknown error")
                st.error(f"❌ Failed to submit: {err}")

    st.markdown("---")
    st.caption(
        "📌 Feedback is linked to your account and stored securely in Firebase. "
        "We do not share your data with third parties."
    )
    
# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <hr>
    <p style='text-align:center;color:gray;font-size:0.8em'>
    BMTC Delay Prediction · Bengaluru ·
    </p>
""", unsafe_allow_html=True)