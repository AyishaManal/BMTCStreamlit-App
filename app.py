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

tab1, tab2, tab3 = st.tabs(["🔍 Predict Journey",
                              "📊 Model Results",
                              "ℹ️ About Project"])

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

            # ── Sanitise note: convert any markdown links → plain <a> tags ─────
            # find_buses may return notes like "[text](url)" – invalid inside HTML
            def _md_link_to_html(text: str) -> str:
                import re as _re
                return _re.sub(
                    r"\[([^\]]+)\]\((https?://[^\)]+)\)",
                    r'<a href="\2" target="_blank" style="color:#3B82F6">\1</a>',
                    text
                )
            note = _md_link_to_html(note)

            # ── Build bus-number HTML for the unified card ────────────────────
            source_dot = {
                "live"  : ("#22C55E", "🟢 Live data"),
                "gtfs"  : ("#EAB308", "🟡 GTFS offline"),
                "static": ("#F97316", "🟠 Estimated"),
                "none"  : ("#9CA3AF", "⚪ Not found"),
            }.get(source, ("#9CA3AF", ""))
            dot_color, dot_label = source_dot

            if buses:
                bus_pills_html = "".join(
                    f"<span style='"
                    f"display:inline-block;background:#1E3A5F;color:#E0F2FE;"
                    f"border-radius:6px;padding:4px 10px;margin:3px 4px 3px 0;"
                    f"font-weight:700;font-size:0.95rem;letter-spacing:0.03em"
                    f"'>{b}</span>"
                    for b in buses
                )
                bus_section_html = f"""
                <div style='margin-top:14px;padding-top:14px;
                            border-top:1px solid rgba(0,0,0,0.08)'>
                    <p style='margin:0 0 6px 0;font-weight:600;color:#1A3A5C;font-size:0.9rem'>
                        🚌 Buses on this corridor
                    </p>
                    <div>{bus_pills_html}</div>
                    <p style='margin:6px 0 0 0;font-size:0.75rem;color:{dot_color}'>
                        {dot_label} · {note}
                    </p>
                </div>
                """
            else:
                bus_section_html = f"""
                <div style='margin-top:14px;padding-top:14px;
                            border-top:1px solid rgba(0,0,0,0.08)'>
                    <p style='margin:0;font-size:0.85rem;color:#6B7280'>
                        🚌 Bus numbers unavailable for this corridor.<br>
                        <span style='font-size:0.75rem'>{note}</span>
                    </p>
                </div>
                """

            # ── Weather & time meta-line ──────────────────────────────────────
            weather_icon = "🌧️ Rain" if is_rain else "☀️ Clear"
            rush_note    = ""
            if hour in [7, 8, 9]:
                rush_note = " &nbsp;·&nbsp; 🔴 AM Rush"
            elif hour in [17, 18, 19]:
                rush_note = " &nbsp;·&nbsp; 🔴 PM Rush"

            # ── Tip text ─────────────────────────────────────────────────────
            if worse >= 8:
                tip_bg, tip_icon, tip_text = "#FEE2E2", "🔴", "Leave early — heavy delays expected!"
            elif worse >= 3:
                tip_bg, tip_icon, tip_text = "#FEF3C7", "⚠️", "Keep a 10-minute buffer for this journey."
            else:
                tip_bg, tip_icon, tip_text = "#D1FAE5", "✅", "Good time to travel — minimal delays expected."

            # ══════════════════════════════════════════════════════════════════
            # UNIFIED OUTPUT CARD
            # ══════════════════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown(f"""
            <div style='
                background:#FFFFFF;
                border:1px solid #E2E8F0;
                border-radius:14px;
                padding:22px 24px 18px 24px;
                box-shadow:0 2px 12px rgba(0,0,0,0.07);
                margin-top:4px
            '>
                <!-- ── Journey meta header ── -->
                <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>
                    <span style='font-size:1.05rem;font-weight:700;color:#1A3A5C'>
                        📍 {src_stop}
                    </span>
                    <span style='color:#64748B;font-size:1.1rem'>→</span>
                    <span style='font-size:1.05rem;font-weight:700;color:#1A3A5C'>
                        🏁 {dst_stop}
                    </span>
                </div>
                <p style='margin:0 0 16px 0;font-size:0.8rem;color:#64748B'>
                    📅 {travel_date.strftime('%d %b %Y')} ({day})
                    &nbsp;·&nbsp; ⏰ {selected_label}
                    &nbsp;·&nbsp; {weather_icon}{rush_note}
                </p>

                <!-- ── Two delay sub-cards side by side ── -->
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px'>

                    <div style='
                        background:{src_bg};
                        border-left:5px solid {src_color};
                        border-radius:10px;
                        padding:16px 18px
                    '>
                        <p style='margin:0 0 2px 0;font-size:0.75rem;
                                  font-weight:600;color:{src_color};text-transform:uppercase;
                                  letter-spacing:0.06em'>FROM</p>
                        <p style='margin:0 0 6px 0;font-size:0.85rem;
                                  font-weight:600;color:{src_color}'>{src_stop}</p>
                        <p style='margin:0;font-size:2rem;font-weight:800;
                                  color:{src_color};line-height:1'>{src_delay} <span style='font-size:1rem'>min</span></p>
                        <p style='margin:4px 0 0 0;font-size:0.85rem;color:{src_color}'>{src_label}</p>
                    </div>

                    <div style='
                        background:{dst_bg};
                        border-left:5px solid {dst_color};
                        border-radius:10px;
                        padding:16px 18px
                    '>
                        <p style='margin:0 0 2px 0;font-size:0.75rem;
                                  font-weight:600;color:{dst_color};text-transform:uppercase;
                                  letter-spacing:0.06em'>TO</p>
                        <p style='margin:0 0 6px 0;font-size:0.85rem;
                                  font-weight:600;color:{dst_color}'>{dst_stop}</p>
                        <p style='margin:0;font-size:2rem;font-weight:800;
                                  color:{dst_color};line-height:1'>{dst_delay} <span style='font-size:1rem'>min</span></p>
                        <p style='margin:4px 0 0 0;font-size:0.85rem;color:{dst_color}'>{dst_label}</p>
                    </div>

                </div>

                <!-- ── Bus numbers (wired from find_buses) ── -->
                {bus_section_html}

                <!-- ── Travel tip banner ── -->
                <div style='
                    margin-top:14px;
                    background:{tip_bg};
                    border-radius:8px;
                    padding:10px 14px;
                    font-size:0.88rem;
                    font-weight:600;
                    color:#1A3A5C
                '>
                    {tip_icon} {tip_text}
                </div>

            </div>
            """, unsafe_allow_html=True)

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
with tab3:
    st.subheader("About This Project")

    n_stops   = metadata.get("n_stops", "~1,955")
    n_prophet = len(prophet_stops)

    st.markdown(f"""
    **Project:** Real-Time Public Transport Delay Prediction — Bengaluru

    **Domain:** Machine Learning | Time-Series Forecasting | Regression

    **Dataset:** BMTC GTFS Aggregated Data (4,655 real Bengaluru bus stops
    with trip counts, route counts, and GPS coordinates)

    ---

    #### Model Architecture (Hybrid)

    | Model | Type | Scope | Purpose |
    |---|---|---|---|
    | **XGBoost** | ML Regression | All {n_stops:,} stops | Live delay prediction (primary) |
    | LSTM (Bi-directional) | Deep Learning | Busiest stop | Academic comparison |
    | ARIMA | Statistical | Busiest stop | Time-series baseline |
    | SARIMA | Statistical | Busiest stop | Seasonal time-series baseline |
    | Prophet | Time-series | Top {n_prophet} high-delay stops | 24-hr forecast chart |

    ---

    #### Real-Time Features
    - 🌦️ **Live weather** — OpenWeatherMap API fetches current Bengaluru
      rain status every 10 minutes and auto-sets the rain toggle
    - 🕐 **Current time default** — time selector defaults to current
      Bengaluru IST time (Asia/Kolkata timezone), rounded to nearest 30 min
    - 📅 **Calendar date picker** — day and month extracted automatically

    ---

    #### How Prediction Works
    1. User types FROM and TO stop — fuzzy search finds the closest match
    2. Travel date selected from calendar (past dates hidden)
    3. Time selected from 30-min slots — defaults to current IST time
    4. Rain status auto-detected from live weather API (overrideable)
    5. **XGBoost** predicts delay using 19 features
    6. Bus numbers looked up from route database
    7. If stop is in top {n_prophet} high-delay stops, **Prophet**
       provides a 24-hour forecast curve with confidence band

    ---

    #### Tools & Libraries
    Python · XGBoost · Prophet · Scikit-learn · TensorFlow/Keras ·
    Pandas · NumPy · Matplotlib · Streamlit · OpenWeatherMap API · Google Colab
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
    <hr>
    <p style='text-align:center;color:gray;font-size:0.8em'>
    BMTC Delay Prediction · Bengaluru
    </p>
""", unsafe_allow_html=True)