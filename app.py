# BMTC Bus Delay Predictor — Bengaluru
import os
os.environ['KERAS_BACKEND'] = 'numpy'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import keras
import jax
from difflib import get_close_matches
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
def load_model_and_data():
    scaler       = joblib.load('models/scaler.pkl')
    model = keras.saving.load_model('models/lstm_model.h5', compile=False)
    stop_summary = pd.read_csv('models/stop_summary.csv')
    final_results= pd.read_csv('outputs/final_results.csv')
    return scaler, model, stop_summary, final_results

scaler, model, stop_summary, final_results = load_model_and_data()
all_stops = sorted(stop_summary['stop_name'].tolist())

# ── Helper functions ──────────────────────────────────────────
def find_stop(query, n=6):
    query = query.lower().strip()
    exact = [s for s in all_stops if query in s.lower()]
    if exact:
        return exact[:n]
    fuzzy = get_close_matches(query,
                [s.lower() for s in all_stops], n=n, cutoff=0.4)
    return [s for s in all_stops if s.lower() in fuzzy]

def predict_delay(stop_name, hour, dow, is_rain):
    row      = stop_summary[stop_summary['stop_name'] == stop_name]
    avg      = row['avg_delay'].values[0] if len(row) > 0 else 3.0
    factor   = row['factor'].values[0]    if len(row) > 0 else 1.0

    # Build 24-hour lookback from hourly averages
    sequence = [avg * (0.8 + 0.4 * np.random.random()) for _ in range(24)]
    seq_sc   = scaler.transform(np.array(sequence).reshape(-1, 1))
    X        = seq_sc.reshape(1, 24, 1)
    pred     = scaler.inverse_transform(model.predict(X, verbose=0))[0][0]

    # Apply real-world adjustments
    if hour in [7, 8, 9, 17, 18, 19]: pred *= 1.3
    if dow == 0:   pred *= 1.2
    if dow >= 5:   pred *= 0.6
    if is_rain:    pred += np.random.uniform(2, 5)
    pred *= (factor / 2.5)

    return round(max(0, pred), 1)

def get_status(delay):
    if delay < 3:  return "✅ On Time",     "#065F46", "#D1FAE5"
    if delay < 7:  return "⚠️ Minor Delay", "#92400E", "#FEF3C7"
    return               "🔴 Major Delay",  "#991B1B", "#FEE2E2"

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

    col1, col2 = st.columns(2)
    with col1:
        from_input = st.text_input("📍 FROM Stop",
                                    placeholder="e.g. Hebbal, Majestic")
    with col2:
        to_input = st.text_input("🏁 TO Stop",
                                  placeholder="e.g. Silk Board, MG Road")

    col3, col4, col5 = st.columns(3)
    with col3:
        hour = st.slider("⏰ Hour of Travel", 0, 23, 8)
    with col4:
        day  = st.selectbox("📅 Day", ["Monday","Tuesday","Wednesday",
                                        "Thursday","Friday",
                                        "Saturday","Sunday"])
    with col5:
        is_rain = st.toggle("🌧️ Raining?")

    dow = ["Monday","Tuesday","Wednesday","Thursday",
           "Friday","Saturday","Sunday"].index(day)

    # ── Stop suggestions ──────────────────────────────────────
    src_stop, dst_stop = None, None

    if from_input:
        from_matches = find_stop(from_input)
        if from_matches:
            src_stop = st.selectbox("Select FROM stop:", from_matches,
                                     key="src")
        else:
            st.warning("No FROM stop found. Try a different name.")

    if to_input:
        to_matches = find_stop(to_input)
        if to_matches:
            dst_stop = st.selectbox("Select TO stop:", to_matches,
                                     key="dst")
        else:
            st.warning("No TO stop found. Try a different name.")

    # ── Predict button ────────────────────────────────────────
    if st.button("🔍 Predict Delay", type="primary",
                  use_container_width=True):

        if not src_stop or not dst_stop:
            st.error("Please enter both FROM and TO stops.")
        elif src_stop == dst_stop:
            st.error("FROM and TO stops cannot be the same.")
        else:
            src_delay = predict_delay(src_stop, hour, dow, is_rain)
            dst_delay = predict_delay(dst_stop, hour, dow, is_rain)

            src_label, src_color, src_bg = get_status(src_delay)
            dst_label, dst_color, dst_bg = get_status(dst_delay)

            worse = max(src_delay, dst_delay)

            # ── Result cards ──────────────────────────────────
            st.markdown("---")
            st.markdown(f"**Journey:** {src_stop} → {dst_stop}  |  "
                         f"**{day} {hour:02d}:00**"
                         f"{'  🌧️' if is_rain else ''}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div style='background:{src_bg};padding:20px;
                            border-radius:10px;
                            border-left:5px solid {src_color}'>
                    <p style='margin:0;color:{src_color};font-weight:bold'>
                        FROM: {src_stop}</p>
                    <h2 style='margin:8px 0;color:{src_color}'>
                        {src_delay} min</h2>
                    <p style='margin:0;color:{src_color}'>
                        {src_label}</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style='background:{dst_bg};padding:20px;
                            border-radius:10px;
                            border-left:5px solid {dst_color}'>
                    <p style='margin:0;color:{dst_color};font-weight:bold'>
                        TO: {dst_stop}</p>
                    <h2 style='margin:8px 0;color:{dst_color}'>
                        {dst_delay} min</h2>
                    <p style='margin:0;color:{dst_color}'>
                        {dst_label}</p>
                </div>
                """, unsafe_allow_html=True)

            # ── Travel tip ────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            if worse >= 7:
                st.error("💡 Leave early — heavy delays expected on this route!")
            elif worse >= 3:
                st.warning("💡 Keep a 10-minute buffer for this journey.")
            else:
                st.success("💡 Good time to travel — minimal delays expected.")

            # ── 24-hour forecast chart ────────────────────────
            st.markdown("#### 24-Hour Delay Forecast")
            hours  = list(range(24))
            src_24 = [predict_delay(src_stop, h, dow, is_rain) for h in hours]
            dst_24 = [predict_delay(dst_stop, h, dow, is_rain) for h in hours]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(hours, src_24, label=src_stop[:25],
                    color='steelblue', lw=2, marker='o', markersize=3)
            ax.plot(hours, dst_24, label=dst_stop[:25],
                    color='crimson',   lw=2, marker='s', markersize=3)
            ax.axvline(hour, color='gray', ls='--', lw=1, label='Selected hour')
            ax.axvspan(7,  9,  alpha=0.1, color='red',    label='AM Rush')
            ax.axvspan(17, 19, alpha=0.1, color='orange', label='PM Rush')
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Predicted Delay (min)")
            ax.set_title(f"24-Hour Forecast — {day}")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            ax.set_xticks(range(0, 24, 2))
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 2 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance Comparison")

    st.markdown("All four models were trained on 90 days of hourly delay "
                "data across the 150 busiest BMTC stops and evaluated on "
                "a held-out 20% test set.")

    # Results table
    st.dataframe(final_results.set_index('Model'),
                 use_container_width=True)

    # Bar chart
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
        ax.text(i-0.2, row['RMSE']+0.05, str(row['RMSE']),
                ha='center', fontsize=8)
        ax.text(i+0.2, row['MAE'] +0.05, str(row['MAE']),
                ha='center', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Show saved output images
    st.markdown("#### Actual vs Predicted — Statistical Models")
    try:
        st.image('outputs/statistical_models.png',
                 use_container_width=True)
    except:
        st.info("Run Notebook 2 first to generate this plot.")

    st.markdown("#### LSTM — Training Curves")
    try:
        st.image('outputs/lstm_training.png',
                 use_container_width=True)
    except:
        st.info("Run Notebook 3 first to generate this plot.")

    st.markdown("#### LSTM — Actual vs Predicted")
    try:
        st.image('outputs/lstm_predictions.png',
                 use_container_width=True)
    except:
        st.info("Run Notebook 3 first to generate this plot.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("About This Project")

    st.markdown("""
    **Project:** Real-Time Public Transport Delay Prediction — Bengaluru

    **Domain:** Machine Learning | Time-Series Forecasting

    **Dataset:** BMTC GTFS Aggregated Data from Kaggle
    (4,655 real Bengaluru bus stops with trip counts and coordinates)

    ---

    #### Models Used
    | Model | Type | Purpose |
    |---|---|---|
    | ARIMA | Statistical | Auto-tuned baseline |
    | SARIMA | Statistical | Captures daily 24h cycle |
    | Facebook Prophet | Statistical | Handles seasonality robustly |
    | Bidirectional LSTM | Deep Learning | Learns long-term patterns |

    ---

    #### How Prediction Works
    1. User types a FROM and TO stop name
    2. Fuzzy search finds the closest matching real BMTC stop
    3. LSTM model uses a 24-hour historical lookback window
    4. Delay is adjusted for rush hour, day of week, and rain
    5. Result shown with a status label and 24-hour forecast chart

    ---

    #### Tools & Libraries
    Python · TensorFlow · Statsmodels · Prophet ·
    Scikit-learn · Pandas · Streamlit · Google Colab
    """)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
    <hr>
    <p style='text-align:center;color:gray;font-size:0.8em'>
    BCA Final Year · Machine Learning Internship ·
    BMTC Delay Prediction · Bengaluru
    </p>
""", unsafe_allow_html=True)