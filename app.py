"""
Introvert vs Extrovert Classifier — Streamlit App
==================================================
Folder structure expected:
    your_project/
    ├── app.py                  ← this file
    └── model_files/
        ├── xgbc_model.pkl
        ├── feature_columns.json
        ├── label_map.json
        └── medians.json

Run with:
    pip install streamlit xgboost scikit-learn pandas numpy
    streamlit run app.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────
# 0.  Page config — MUST be first st call
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Introvert vs Extrovert Classifier",
    page_icon="🧠",
    layout="centered",
)

# ─────────────────────────────────────────
# 1.  Load model artifacts
# ─────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "model_files"

@st.cache_resource
def load_artifacts():
    with open(MODEL_DIR / "xgbc_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    with open(MODEL_DIR / "label_map.json") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    with open(MODEL_DIR / "medians.json") as f:
        medians = json.load(f)
    return model, feature_columns, label_map, medians

model, FEATURE_COLUMNS, LABEL_MAP, MEDIANS = load_artifacts()


# ─────────────────────────────────────────
# 2.  Feature engineering (mirrors training)
# ─────────────────────────────────────────
def build_features(raw: dict) -> pd.DataFrame:
    """Apply the same encoding + feature engineering as in the notebook."""

    # -- binary encoding --
    stage_fear           = 1 if raw["Stage_fear"] == "Yes" else 0
    drained_after_social = 1 if raw["Drained_after_socializing"] == "Yes" else 0

    # -- base features --
    time_alone     = raw["Time_spent_Alone"]
    social_events  = raw["Social_event_attendance"]
    going_outside  = raw["Going_outside"]
    friends        = raw["Friends_circle_size"]
    post_freq      = raw["Post_frequency"]

    # -- engineered features (exact same formulas as notebook) --
    social_engagement = social_events + going_outside + post_freq
    introvert_score   = time_alone + stage_fear + drained_after_social
    social_vs_alone   = social_engagement / (time_alone + 1)
    friends_per_event = friends / (social_events + 1)
    online_vs_offline = post_freq / (going_outside + 1)

    # -- binned feature --
    bins   = [0, 2, 5, 8, 10]
    labels = [0, 1, 2, 3]
    alone_time_level = int(pd.cut([time_alone], bins=bins, labels=labels)[0])

    row = {
        "Time_spent_Alone":        time_alone,
        "Stage_fear":              stage_fear,
        "Social_event_attendance": social_events,
        "Going_outside":           going_outside,
        "Drained_after_socializing": drained_after_social,
        "Friends_circle_size":     friends,
        "Post_frequency":          post_freq,
        "social_engagement":       social_engagement,
        "introvert_score":         introvert_score,
        "social_vs_alone":         social_vs_alone,
        "friends_per_event":       friends_per_event,
        "online_vs_offline":       online_vs_offline,
        "alone_time_level":        alone_time_level,
    }

    df = pd.DataFrame([row])[FEATURE_COLUMNS]   # guarantee column order
    return df


# ─────────────────────────────────────────
# 3.  Layout
# ─────────────────────────────────────────
st.title("🧠 Introvert vs Extrovert Classifier")
st.markdown(
    "Fill in the details below and the XGBoost model will predict your personality type."
)

st.divider()

# ─────────────────────────────────────────
# 4.  Input form
# ─────────────────────────────────────────
with st.form("personality_form"):
    st.subheader("🙋 Tell us about yourself")

    col1, col2 = st.columns(2)

    with col1:
        time_alone = st.number_input(
            "⏱ Time Spent Alone (hrs/day, 0–10)",
            min_value=0.0, max_value=10.0,
            value=float(MEDIANS["Time_spent_Alone"]),
            step=0.5,
        )

        stage_fear = st.selectbox(
            "🎤 Stage Fear",
            options=["No", "Yes"],
        )

        social_events = st.number_input(
            "🎉 Social Event Attendance (0–10)",
            min_value=0.0, max_value=10.0,
            value=float(MEDIANS["Social_event_attendance"]),
            step=0.5,
        )

        going_outside = st.number_input(
            "🚶 Going Outside (times/week, 0–10)",
            min_value=0.0, max_value=10.0,
            value=float(MEDIANS["Going_outside"]),
            step=0.5,
        )

    with col2:
        drained = st.selectbox(
            "😓 Drained After Socializing",
            options=["No", "Yes"],
        )

        friends = st.number_input(
            "👥 Friends Circle Size (0–20)",
            min_value=0.0, max_value=20.0,
            value=float(MEDIANS["Friends_circle_size"]),
            step=1.0,
        )

        post_freq = st.number_input(
            "📱 Post Frequency (posts/week, 0–10)",
            min_value=0.0, max_value=10.0,
            value=float(MEDIANS["Post_frequency"]),
            step=0.5,
        )

    submitted = st.form_submit_button("🔮 Predict My Personality", use_container_width=True)


# ─────────────────────────────────────────
# 5.  Prediction & result display
# ─────────────────────────────────────────
if submitted:
    raw_input = {
        "Time_spent_Alone":        time_alone,
        "Stage_fear":              stage_fear,
        "Social_event_attendance": social_events,
        "Going_outside":           going_outside,
        "Drained_after_socializing": drained,
        "Friends_circle_size":     friends,
        "Post_frequency":          post_freq,
    }

    features   = build_features(raw_input)
    prediction = int(model.predict(features)[0])
    proba      = model.predict_proba(features)[0]     # [P(Introvert), P(Extrovert)]
    label      = LABEL_MAP[prediction]
    confidence = proba[prediction] * 100

    st.divider()
    st.subheader("🎯 Prediction Result")

    if label == "Introvert":
        st.success(f"### 🌙 You are likely an **Introvert** ({confidence:.1f}% confidence)")
        st.markdown(
            "Introverts tend to recharge through solitude, prefer deeper one-on-one "
            "conversations, and may feel drained after extensive social interaction."
        )
    else:
        st.success(f"### ☀️ You are likely an **Extrovert** ({confidence:.1f}% confidence)")
        st.markdown(
            "Extroverts tend to gain energy from social interaction, enjoy large gatherings, "
            "and are generally comfortable in the spotlight."
        )

    # -- probability bar chart --
    st.markdown("#### 📊 Confidence Breakdown")
    proba_df = pd.DataFrame(
        {"Personality": ["Introvert", "Extrovert"], "Probability": [proba[0], proba[1]]}
    ).set_index("Personality")
    st.bar_chart(proba_df)

    # -- engineered features for transparency --
    with st.expander("🔍 Engineered features used by the model"):
        features_display = features.copy()
        features_display.index = ["Your values"]
        st.dataframe(features_display.T.rename(columns={"Your values": "Value"}))

st.divider()
st.caption("Model: XGBoost (GridSearchCV-tuned) · Built with Streamlit")