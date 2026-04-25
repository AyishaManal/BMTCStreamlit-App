# BMTC Bus Delay Predictor — Bengaluru

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from difflib import get_close_matches
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="BMTC Delay Predictor",
    page_icon="🚌",
    layout="centered"
)

# ── Load saved files ──────────────────────────────────────────
@st.cache_resource
def load_assets():
    scaler       = joblib.load('models/scaler.pkl')
    model        = tf.keras.models.load_model('models/lstm_model.h5',
                                               compile=False)
    stop_summary = pd.read_csv('models/stop_summary.csv')
    final_results= pd.read_csv('outputs/final_results.csv')
    routes       = pd.read_csv('models/routes.csv')
    return scaler, model, stop_summary, final_results, routes

scaler, model, stop_summary, final_results, routes_df = load_assets()
all_stops = sorted(stop_summary['stop_name'].tolist())

# ── Helper: fuzzy stop search ─────────────────────────────────
def find_stop(query, n=6):
    query = query.lower().strip()
    exact = [s for s in all_stops if query in s.lower()]
    if exact:
        return exact[:n]
    fuzzy = get_close_matches(
        query, [s.lower() for s in all_stops], n=n, cutoff=0.4)
    return [s for s in all_stops if s.lower() in fuzzy]

# ── Helper: find bus numbers for a stop pair ──────────────────
def find_buses(src, dst):
    """
    Looks up route_stops.csv to find bus numbers
    that serve both the FROM and TO stops.
    Returns a list of bus numbers or empty list.
    """
    src_lower = src.lower()
    dst_lower = dst.lower()

    matched = []
    for _, row in routes_df.iterrows():
        stops_on_route = str(row['stops']).lower()
        if src_lower[:8] in stops_on_route and dst_lower[:8] in stops_on_route:
            matched.append(str(row['bus_number']))

    return matched[:5]  # return max 5 bus numbers

# ── Helper: predict delay ─────────────────────────────────────
def predict_delay(stop_name, hour, dow, is_rain, month):
    row    = stop_summary[stop_summary['stop_name'] == stop_name]
    avg    = row['avg_delay'].values[0] if len(row) > 0 else 3.0
    factor = row['factor'].values[0]    if len(row) > 0 else 1.0

    sequence = [avg * (0.8 + 0.4 * np.random.random()) for _ in range(24)]
    seq_sc   = scaler.transform(np.array(sequence).reshape(-1, 1))
    X        = seq_sc.reshape(1, 24, 1)
    pred     = scaler.inverse_transform(
                   model.predict(X, verbose=0))[0][0]

    # Rush hour
    if hour in [7, 8, 9, 17, 18, 19]: pred *= 1.3
    # Day of week
    if dow == 0:  pred *= 1.2   # Monday
    if dow >= 5:  pred *= 0.6   # Weekend
    # Rain
    if is_rain:   pred += np.random.uniform(2, 5)
    # Monsoon months (June–September in Bengaluru)
    if month in [6, 7, 8, 9]:  pred *= 1.15
    # Congestion factor from real data
    pred *= (factor / 2.5)

    return round(max(0, pred), 1)

def get_status(delay):
    if delay < 3:  return "✅ On Time",      "#065F46", "#D1FAE5"
    if delay < 7:  return "⚠️ Minor Delay",  "#92400E", "#FEF3C7"
    return               "🔴 Major Delay",   "#991B1B", "#FEE2E2"

# ── Header ────────────────────────────────────────────────────
st.markdown("""
    <h1 style='color:#1A3A5C; margin-bottom:0'>🚌 BMTC Delay Predictor</h1>
    <p style='color:gray; margin-top:4px'>
        Bengaluru · ML-Powered Bus Delay Forecasting ·
        BCA Final Year Internship Project
    </p>
    <hr>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict Journey",
                              "📊 Model Results",
                              "ℹ️ About Project"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — JOURNEY PREDICTOR
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Your Journey Details")

    # ── Stop inputs ───────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("📍 **FROM Stop**")
        from_input = st.text_input(" ", placeholder="e.g. Hebbal, Majestic",
                                    label_visibility="collapsed",
                                    key="from_input")
    with col2:
        st.markdown("🏁 **TO Stop**")
        to_input = st.text_input(" ", placeholder="e.g. Silk Board, MG Road",
                                  label_visibility="collapsed",
                                  key="to_input")

    # ── Stop selectors (appear after typing) ──────────────────
    src_stop, dst_stop = None, None

    if from_input:
        from_matches = find_stop(from_input)
        if from_matches:
            src_stop = st.selectbox("Select FROM stop:", from_matches,
                                     key="src_sel")
        else:
            st.warning("No stop found. Try: Hebbal, Silk Board, Majestic, MG Road")

    if to_input:
        to_matches = find_stop(to_input)
        if to_matches:
            dst_stop = st.selectbox("Select TO stop:", to_matches,
                                     key="dst_sel")
        else:
            st.warning("No stop found. Try: Indiranagar, Koramangala, Whitefield")

    st.markdown("---")

    # ── Date picker (calendar — no past dates) ────────────────
    st.markdown("📅 **Select Travel Date**")
    today    = date.today()
    max_date = today + timedelta(days=30)   # allow 30 days ahead

    travel_date = st.date_input(
        " ",
        value=today,
        min_value=today,        # hides all past dates
        max_value=max_date,
        label_visibility="collapsed"
    )

    # Extract day of week and month from selected date
    dow   = travel_date.weekday()   # 0=Monday, 6=Sunday
    month = travel_date.month       # 1–12

    day_name = ["Monday","Tuesday","Wednesday","Thursday",
                "Friday","Saturday","Sunday"][dow]
    month_name = travel_date.strftime("%B")   # e.g. "June"

    st.caption(f"Selected: **{travel_date.strftime('%d %B %Y')}** "
               f"({day_name})  —  Month: {month_name}")

    # ── Hour slider ───────────────────────────────────────────
    st.markdown("⏰ **Hour of Travel**")
    hour = st.slider(" ", 0, 23, 8,
                     format="%d:00",
                     label_visibility="collapsed")

    # Show what time period this is
    if hour in [7, 8, 9]:
        st.caption("🔴 AM Rush Hour — expect higher delays")
    elif hour in [17, 18, 19]:
        st.caption("🔴 PM Rush Hour — expect higher delays")
    elif 0 <= hour <= 5:
        st.caption("🌙 Late Night — minimal traffic")
    else:
        st.caption("🟡 Normal Hours")

    # ── Rain toggle ───────────────────────────────────────────
    is_rain = st.toggle("🌧️  Raining?")
    if month in [6, 7, 8, 9]:
        st.caption("☔ Monsoon season — rain likely to add 2–5 min delay")

    st.markdown("---")

    # ── Predict button ────────────────────────────────────────
    predict_btn = st.button("🔍 Predict Delay",
                             type="primary",
                             use_container_width=True)

    if predict_btn:
        if not src_stop or not dst_stop:
            st.error("Please enter and select both FROM and TO stops.")
        elif src_stop == dst_stop:
            st.error("FROM and TO stops cannot be the same.")
        else:
            src_delay = predict_delay(src_stop, hour, dow, is_rain, month)
            dst_delay = predict_delay(dst_stop, hour, dow, is_rain, month)

            src_label, src_color, src_bg = get_status(src_delay)
            dst_label, dst_color, dst_bg = get_status(dst_delay)
            worse = max(src_delay, dst_delay)

            # ── Journey header ────────────────────────────────
            st.markdown("---")
            st.markdown(
                f"**{src_stop}** → **{dst_stop}**  |  "
                f"{travel_date.strftime('%d %b %Y')} ({day_name})  |  "
                f"{hour:02d}:00"
                f"{'  🌧️ Rain' if is_rain else ''}"
            )

            # ── Bus numbers ───────────────────────────────────
            buses = find_buses(src_stop, dst_stop)
            if buses:
                st.markdown(
                    f"🚌 **Bus Numbers Available:** `{'`  `'.join(buses)}`"
                )
            else:
                # Friendly fallback if no exact match found
                st.markdown(
                    "🚌 **Bus Numbers:** Check BMTC app for exact bus numbers "
                    "on this route — multiple services may be available."
                )

            # ── Delay result cards ────────────────────────────
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div style='background:{src_bg};padding:20px;
                            border-radius:12px;
                            border-left:5px solid {src_color};
                            margin-top:12px'>
                    <p style='margin:0;color:{src_color};
                              font-weight:bold;font-size:0.9em'>
                        FROM</p>
                    <p style='margin:4px 0;color:{src_color};
                              font-weight:bold'>
                        {src_stop}</p>
                    <h2 style='margin:8px 0;color:{src_color}'>
                        {src_delay} min</h2>
                    <p style='margin:0;color:{src_color}'>
                        {src_label}</p>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div style='background:{dst_bg};padding:20px;
                            border-radius:12px;
                            border-left:5px solid {dst_color};
                            margin-top:12px'>
                    <p style='margin:0;color:{dst_color};
                              font-weight:bold;font-size:0.9em'>
                        TO</p>
                    <p style='margin:4px 0;color:{dst_color};
                              font-weight:bold'>
                        {dst_stop}</p>
                    <h2 style='margin:8px 0;color:{dst_color}'>
                        {dst_delay} min</h2>
                    <p style='margin:0;color:{dst_color}'>
                        {dst_label}</p>
                </div>
                """, unsafe_allow_html=True)

            # ── Travel tip ────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            if worse >= 7:
                st.error(
                    "💡 Leave early — heavy delays expected. "
                    "Consider the next bus or an alternate route.")
            elif worse >= 3:
                st.warning("💡 Keep a 10-minute buffer for this journey.")
            else:
                st.success("💡 Good time to travel — minimal delays expected.")

            # ── 24-hour forecast chart ────────────────────────
            st.markdown("#### 24-Hour Delay Forecast")
            hours  = list(range(24))
            np.random.seed(42)
            src_24 = [predict_delay(src_stop, h, dow, is_rain, month)
                      for h in hours]
            np.random.seed(99)
            dst_24 = [predict_delay(dst_stop, h, dow, is_rain, month)
                      for h in hours]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(hours, src_24, label=src_stop[:20],
                    color='steelblue', lw=2, marker='o', markersize=3)
            ax.plot(hours, dst_24, label=dst_stop[:20],
                    color='crimson', lw=2, marker='s', markersize=3)
            ax.axvline(hour, color='gray', ls='--', lw=1.2,
                       label=f'Your hour ({hour}:00)')
            ax.axvspan(7,  9,  alpha=0.08, color='red',
                       label='AM Rush')
            ax.axvspan(17, 19, alpha=0.08, color='orange',
                       label='PM Rush')
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Predicted Delay (min)")
            ax.set_title(
                f"{day_name}, {travel_date.strftime('%d %b %Y')}"
                f"{'  🌧️' if is_rain else ''}"
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_xticks(range(0, 24, 2))
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 2 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance Comparison")
    st.markdown("Trained on 90 days of hourly data across "
                "150 real BMTC stops. Evaluated on 20% held-out test set.")

    st.dataframe(final_results.set_index('Model'),
                 use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(final_results))
    ax.bar(x - 0.2, final_results['RMSE'], 0.35,
           label='RMSE', color='steelblue')
    ax.bar(x + 0.2, final_results['MAE'],  0.35,
           label='MAE',  color='darkorange')
    ax.set_xticks(x)
    ax.set_xticklabels(final_results['Model'], rotation=10)
    ax.set_ylabel('Error (minutes)')
    ax.set_title('RMSE & MAE — All Models')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    for i, row in final_results.iterrows():
        ax.text(i-0.2, row['RMSE']+0.03, str(row['RMSE']),
                ha='center', fontsize=8)
        ax.text(i+0.2, row['MAE'] +0.03, str(row['MAE']),
                ha='center', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Statistical Models — Actual vs Predicted")
    try:
        st.image('outputs/statistical_models.png',
                 use_container_width=True)
    except:
        st.info("Run Notebook 2 to generate this plot.")

    st.markdown("#### LSTM — Training Loss Curves")
    try:
        st.image('outputs/lstm_training.png',
                 use_container_width=True)
    except:
        st.info("Run Notebook 3 to generate this plot.")

    st.markdown("#### LSTM — Actual vs Predicted")
    try:
        st.image('outputs/lstm_predictions.png',
                 use_container_width=True)
    except:
        st.info("Run Notebook 3 to generate this plot.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("About This Project")
    st.markdown("""
    **Project:** Real-Time Public Transport Delay Prediction — Bengaluru

    **Domain:** Machine Learning | Time-Series Forecasting

    **Dataset:** BMTC GTFS Aggregated Data (Kaggle)
    — 4,655 real Bengaluru bus stops with trip counts and coordinates

    ---

    #### Models Used
    | Model | Type | Purpose |
    |---|---|---|
    | ARIMA | Statistical | Auto-tuned baseline |
    | SARIMA | Statistical | Captures 24-hour daily cycle |
    | Facebook Prophet | Statistical | Handles seasonality robustly |
    | Bidirectional LSTM | Deep Learning | Learns sequential patterns |

    ---

    #### How Prediction Works
    1. User selects FROM and TO stop using fuzzy search
    2. User picks a travel date from the calendar (no past dates)
    3. Day of week and month are extracted automatically
    4. LSTM uses a 24-hour lookback window to predict delay
    5. Monsoon months (June–Sep) apply an extra delay factor
    6. Bus numbers are looked up from the route database
    7. Result displayed with status card and 24-hour chart

    ---

    #### Tools
    Python · TensorFlow · Statsmodels · Prophet ·
    Scikit-learn · Pandas · Streamlit · Google Colab
    """)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
    <hr>
    <p style='text-align:center;color:gray;font-size:0.8em'>
     Machine Learning Internship ·
    BMTC Delay Prediction · Bengaluru
    </p>
""", unsafe_allow_html=True)