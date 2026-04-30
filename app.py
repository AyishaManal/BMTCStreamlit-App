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
# Add it to Streamlit Cloud: Settings → Secrets → paste below
# [openweather]

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

@st.cache_data
def load_routes():
    path = os.path.join(MODEL_DIR, "routes.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[["name", "full_name", "trip_count"]].copy()
        df.columns = ["bus_number", "full_name", "trip_count"]
        df["full_name_lower"] = df["full_name"].str.lower().fillna("")
        return df
    return pd.DataFrame(columns=["bus_number", "full_name",
                                  "trip_count", "full_name_lower"])

# ── Load everything at startup ────────────────────────────────────────────────
try:
    xgb_model, stop_summary, metadata = load_assets()
    final_results = load_results()
    prophet_stops = load_prophet_stops()
    routes_df     = load_routes()
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
 
def find_buses(src, dst):
    """
    Find bus numbers connecting src and dst stops.
 
    Strategy:
      - Extract the core AREA keyword from each stop name by stripping common
        suffixes (Police Station, Bus Station, Gate, etc.) that appear in
        stop_summary but NOT in routes.csv full_names.
      - Score each route by how many area keywords it contains from each side.
      - Confidence 3 → both src AND dst area found in route name (best match).
      - Confidence 2 → only src area found (buses serving the FROM area).
      - Confidence 1 → only dst area found (buses serving the TO area, fallback).
      - Deduplicate by bus number (same route appears in both directions).
      - Return top-5 by (confidence DESC, trip_count DESC).
    """
    if routes_df.empty:
        return []
 
    def area_keywords(stop_name):
        """Strip common stop-type suffixes to get the bare area/locality name."""
        suffixes = [
            "police station", "bus station", "bus stand", "bus stop",
            "railway station", "metro station", "metro", "circle", "gate",
            "junction", "flyover", "bridge", "hospital", "school", "college",
            "depot", "terminal",
        ]
        name = stop_name.lower()
        for s in suffixes:
            name = name.replace(s, "").strip()
        skip = {
            "bus", "stop", "the", "and", "for", "near", "road", "main",
            "cross", "layout", "nagar", "stage", "old", "new", "town", "cs-",
        }
        words = name.replace("-", " ").split()
        keys = [w for w in words if len(w) > 2 and w not in skip]
        return keys
 
    src_keys = area_keywords(src)
    dst_keys = area_keywords(dst)
 
    # Score every route; keep best (confidence, trip_count) per bus number
    bus_best = {}   # bus_number → (confidence, trip_count, full_name)
    for _, row in routes_df.iterrows():
        text = row["full_name_lower"]
        src_score = sum(1 for k in src_keys if k in text)
        dst_score = sum(1 for k in dst_keys if k in text)
 
        if src_score >= 1 and dst_score >= 1:
            confidence = 3
        elif src_score >= 1:
            confidence = 2
        elif dst_score >= 1:
            confidence = 1
        else:
            continue
 
        bus = row["bus_number"]
        tc  = row["trip_count"]
        if bus not in bus_best or (confidence, tc) > (bus_best[bus][0], bus_best[bus][1]):
            bus_best[bus] = (confidence, tc, row["full_name"])
 
    if not bus_best:
        return []
 
    sorted_buses = sorted(
        bus_best.items(),
        key=lambda x: (x[1][0], x[1][1]),   # sort by confidence then trip_count
        reverse=True,
    )
 
    # Prefer confidence-3 routes; if none, return confidence-2 with a note
    top = sorted_buses[:5]
    return [(bus, conf, fn) for bus, (conf, tc, fn) in top]
 
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
            src_delay = predict_delay(src_stop, hour, dow, month, is_rain)
            dst_delay = predict_delay(dst_stop, hour, dow, month, is_rain)

            src_label, src_color, src_bg = get_status(src_delay)
            dst_label, dst_color, dst_bg = get_status(dst_delay)
            worse = max(src_delay, dst_delay)

            # ── Journey header ────────────────────────────────────────────────
            st.markdown("---")
            weather_note = f"🌧️ Rain" if is_rain else "☀️ Clear"
            st.markdown(
                f"**Journey:** {src_stop} → {dst_stop} &nbsp;|&nbsp; "
                f"**{travel_date.strftime('%d %b %Y')} ({day}) "
                f"{selected_label}** &nbsp;|&nbsp; {weather_note}"
            )

            # ── Bus numbers ───────────────────────────────────────────────────
            buses = find_buses(src_stop, dst_stop)
            if buses:
                bus_tags = "  ".join([f"`{b}`" for b in buses])
                st.markdown(f"🚌 **Bus Numbers:** {bus_tags}")
            else:
                st.markdown(
                    "🚌 **Bus Numbers:** Multiple BMTC services available — "
                    "check the BMTC app for exact bus numbers on this corridor."
                )

            # ── Delay result cards ────────────────────────────────────────────
            c1, c2 = st.columns(2)
            for col, stop, delay, label, color, bg in [
                (c1, src_stop, src_delay, src_label, src_color, src_bg),
                (c2, dst_stop, dst_delay, dst_label, dst_color, dst_bg),
            ]:
                with col:
                    role = "FROM" if col == c1 else "TO"
                    st.markdown(f"""
                    <div style='background:{bg};padding:20px;border-radius:10px;
                                border-left:5px solid {color};margin-top:10px'>
                        <p style='margin:0;color:{color};font-weight:bold'>
                            {role}: {stop}</p>
                        <h2 style='margin:8px 0;color:{color}'>{delay} min</h2>
                        <p style='margin:0;color:{color}'>{label}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Travel tip ────────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            if worse >= 8:
                st.error("💡 Leave early — heavy delays expected on this route!")
            elif worse >= 3:
                st.warning("💡 Keep a 10-minute buffer for this journey.")
            else:
                st.success("💡 Good time to travel — minimal delays expected.")

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