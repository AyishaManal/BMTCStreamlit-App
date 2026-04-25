# BMTC Bus Delay Predictor — Bengaluru
# Option C: XGBoost (all stops) + Prophet (top 30 high-delay stops)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os
from difflib import get_close_matches
from datetime import date, timedelta          # ← NEW: for calendar
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMTC Delay Predictor",
    page_icon="🚌",
    layout="centered"
)

# ── File paths — all relative to repo root ────────────────────────────────────
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"

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

# ── Load real BMTC routes dataset (routes.csv) ───────────────────────────────
@st.cache_data
def load_routes():
    # Columns in routes.csv:
    #   name        → bus number  e.g. "500C", "V-500D", "335E"
    #   full_name   → "Source Stop → Destination Stop"
    #   trip_count  → how many daily trips on this route
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
    routes_df     = load_routes()           # ← NEW
    FEATURES      = metadata["features"]
    all_stops     = sorted(stop_summary["stop_name"].tolist())
except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.info(
        "Make sure you have run all three notebooks and copied the "
        "`models/` and `outputs/` folders into the root of your GitHub repo."
    )
    st.stop()

# ── Helper: fuzzy stop search ─────────────────────────────────────────────────
def find_stop(query, n=6):
    query = query.lower().strip()
    exact = [s for s in all_stops if query in s.lower()]
    if exact:
        return exact[:n]
    fuzzy = get_close_matches(query, [s.lower() for s in all_stops],
                              n=n, cutoff=0.4)
    return [s for s in all_stops if s.lower() in fuzzy]

# ── Find bus numbers using real routes.csv data ───────────────────────────────
def find_buses(src, dst):
    """
    Uses the real BMTC routes.csv dataset.
    full_name column has "Source → Destination" format.
    We search for routes where full_name contains
    keywords from both src and dst stop names.
    Returns up to 5 matching bus numbers sorted by trip_count.
    """
    if routes_df.empty:
        return []

    # Use first meaningful keyword from each stop name
    # e.g. "Kempegowda Bus Station" → "kempegowda"
    #      "Central Silk Board"     → "silk"
    def keywords(stop_name):
        skip = {"bus","station","stop","road","gate","circle",
                "nagar","layout","cross","main","bridge","town"}
        words = stop_name.lower().split()
        keys  = [w for w in words if len(w) > 3 and w not in skip]
        return keys[:2] if keys else [stop_name.lower()[:6]]

    src_keys = keywords(src)
    dst_keys = keywords(dst)

    matched = []
    for _, row in routes_df.iterrows():
        text = row["full_name_lower"]
        src_found = any(k in text for k in src_keys)
        dst_found = any(k in text for k in dst_keys)
        if src_found and dst_found:
            matched.append((row["bus_number"], row["trip_count"]))

    if not matched:
        # Fallback: single-side match on destination only
        for _, row in routes_df.iterrows():
            text = row["full_name_lower"]
            if any(k in text for k in dst_keys):
                matched.append((row["bus_number"], row["trip_count"]))

    # Sort by trip_count descending (busiest routes first)
    matched_sorted = sorted(matched, key=lambda x: x[1], reverse=True)
    return [bus for bus, _ in matched_sorted[:5]]

# ── Helper: build XGBoost feature vector ─────────────────────────────────────
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
        BCA Final Year Internship Project
    </p>
    <hr>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
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
                                    placeholder="e.g. Indiranagara, Silk Board")

    # ── CHANGED: Calendar picker replaces Day + Month dropdowns ───────────────
    st.markdown("📅 **Select Travel Date**")
    today    = date.today()
    max_date = today + timedelta(days=30)   # allow up to 30 days ahead

    travel_date = st.date_input(
        " ",
        value=today,
        min_value=today,       # past dates hidden automatically
        max_value=max_date,
        label_visibility="collapsed"
    )

    # Extract dow and month automatically from selected date
    dow   = travel_date.weekday()    # 0=Monday … 6=Sunday
    month = travel_date.month        # 1–12
    day   = ["Monday","Tuesday","Wednesday","Thursday",
             "Friday","Saturday","Sunday"][dow]

    st.caption(
        f"📆 {travel_date.strftime('%d %B %Y')}  ({day})  "
        f"— Month: {travel_date.strftime('%B')}"
    )

    # ── Hour slider ───────────────────────────────────────────────────────────
    hour = st.slider("⏰ Hour of Travel", 0, 23, 8)

    # Show time-period label below the slider
    if hour in [7, 8, 9]:
        st.caption("🔴 AM Rush Hour — expect higher delays")
    elif hour in [17, 18, 19]:
        st.caption("🔴 PM Rush Hour — expect higher delays")
    elif 0 <= hour <= 5:
        st.caption("🌙 Late Night — minimal traffic expected")
    else:
        st.caption("🟡 Normal Hours")

    # ── Rain toggle ───────────────────────────────────────────────────────────
    is_rain = st.toggle("🌧️ Raining?")
    if month in [6, 7, 8, 9]:
        st.caption("☔ Monsoon season — rain likely to add 2–5 min extra delay")

    # ── Stop search results ───────────────────────────────────────────────────
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
            st.markdown(
                f"**Journey:** {src_stop} → {dst_stop} &nbsp;|&nbsp; "
                f"**{travel_date.strftime('%d %b %Y')} ({day}) "
                f"{hour:02d}:00** {'🌧️' if is_rain else ''}"
            )

            # ── NEW: Bus numbers ──────────────────────────────────────────────
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
            hours = list(range(24))

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

            ax.axvline(hour, color='gray', ls='--', lw=1,
                       label=f'Your hour ({hour:02d}:00)')
            ax.axvspan(7,  9,  alpha=0.10, color='red',    label='AM Rush')
            ax.axvspan(17, 19, alpha=0.10, color='orange', label='PM Rush')
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Predicted Delay (min)")
            ax.set_title(
                f"{travel_date.strftime('%d %b %Y')} ({day})"
                f"{'  🌧️' if is_rain else ''}"
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

    #### How Prediction Works
    1. User types FROM and TO stop — fuzzy search finds the closest match
    2. User selects travel date from calendar (past dates hidden)
    3. Day of week and month are extracted automatically from the date
    4. **XGBoost** predicts delay using 19 features including congestion
       factor, hour, day, month, rush-hour flag, rain flag, and lag features
    5. Bus numbers are looked up from the route database
    6. If the stop is in the top {n_prophet} high-delay stops, **Prophet**
       provides a 24-hour forecast curve with confidence band
    7. Result shown with status badge, bus numbers, and 24-hour chart

    ---

    #### Tools & Libraries
    Python · XGBoost · Prophet · Scikit-learn · TensorFlow/Keras ·
    Pandas · NumPy · Matplotlib · Streamlit · Google Colab
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
    <hr>
    <p style='text-align:center;color:gray;font-size:0.8em'>
    BCA Final Year · Machine Learning Internship ·
    BMTC Delay Prediction · Bengaluru
    </p>
""", unsafe_allow_html=True)
