import pickle
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

MODEL_PATH = Path(__file__).parent / "diabetes_prediction.pkl"

SMOKING_MAP = {
    "No Info": 0,
    "Current": 1,
    "Ever": 2,
    "Former": 3,
    "Never": 4,
    "Not Current": 5,
}

FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

st.markdown(
    """
<style>
#MainMenu, footer { visibility: hidden; }
.stApp { background: linear-gradient(160deg, #eaf4fb 0%, #f4f8fb 45%, #e9f7f3 100%); }
.block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 2rem; }

.hero {
    background: linear-gradient(120deg, #0f766e, #0e7490, #1d4ed8);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 30px rgba(15, 118, 110, 0.25);
}
.hero h1 { margin: 0; color: #ffffff; font-size: 2rem; }
.hero p { margin: .4rem 0 0 0; color: #ffffff; opacity: .92; }

[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #dbe7f0;
    border-radius: 18px;
    padding: 1.6rem;
    box-shadow: 0 8px 24px rgba(18, 38, 58, 0.08);
}
label, .stMarkdown p { color: #12263a !important; font-weight: 500 !important; }

.section-title {
    color: #0f3d56;
    font-weight: 600;
    font-size: 1.05rem;
    margin: 0 0 .6rem 0;
    padding-bottom: .35rem;
    border-bottom: 3px solid #14b8a6;
    width: fit-content;
}

[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    background: linear-gradient(90deg, #0f766e, #0e7490);
    color: #ffffff;
    font-weight: 600;
    border: none;
    border-radius: 12px;
    padding: .7rem 1rem;
}
[data-testid="stFormSubmitButton"] > button:hover { color: #ffffff; border: none; }

.panel {
    border-radius: 18px;
    padding: 2rem 1.5rem;
    text-align: center;
    min-height: 260px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    box-shadow: 0 8px 24px rgba(18, 38, 58, 0.1);
}
.panel.empty { background: #ffffff; border: 2px dashed #b6cbd9; color: #5b7083; }
.panel.low { background: linear-gradient(135deg, #059669, #10b981); color: #ffffff; }
.panel.high { background: linear-gradient(135deg, #be123c, #e11d48); color: #ffffff; }
.panel .icon { font-size: 3rem; }
.panel h2 { margin: .4rem 0 .2rem 0; color: inherit; font-size: 1.5rem; }
.panel p { margin: 0; color: inherit; }
.panel .bar { width: 100%; height: 12px; background: rgba(255,255,255,.35); border-radius: 999px; margin-top: 1.2rem; overflow: hidden; }
.panel .bar div { height: 100%; background: #ffffff; border-radius: 999px; }

.disclaimer {
    margin-top: 1.5rem;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #7c2d12;
    border-radius: 12px;
    padding: .8rem 1rem;
    font-size: .85rem;
}
</style>
<div class="hero">
    <h1>🩺 Diabetes Risk Predictor</h1>
    <p>Enter your health details on the left to get an instant prediction.</p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load diabetes_prediction.pkl (needs scikit-learn==1.6.1): {e}")
    st.stop()

left, right = st.columns([3, 2], gap="large")

with left:
    with st.form("patient_form"):
        st.markdown('<div class="section-title">👤 Personal details</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        age = a.number_input("Age (years)", min_value=1.0, max_value=100.0, value=40.0, step=1.0)
        bmi = b.number_input("BMI (kg/m²)", min_value=10.0, max_value=95.0, value=25.0, step=0.1)
        smoking = c.selectbox("Smoking history", list(SMOKING_MAP.keys()), index=4)

        st.markdown('<div class="section-title">🩸 Medical details</div>', unsafe_allow_html=True)
        d, e = st.columns(2)
        hba1c = d.number_input("HbA1c level (%)", min_value=3.5, max_value=9.0, value=5.5, step=0.1)
        glucose = e.number_input("Blood glucose level (mg/dL)", min_value=40, max_value=1000, value=120, step=1)

        f, g = st.columns(2)
        hypertension = f.radio("Hypertension", ["No", "Yes"], horizontal=True)
        heart_disease = g.radio("Heart disease", ["No", "Yes"], horizontal=True)

        submitted = st.form_submit_button("Predict")

with right:
    if submitted:
        row = pd.DataFrame(
            [[
                age,
                1 if hypertension == "Yes" else 0,
                1 if heart_disease == "Yes" else 0,
                SMOKING_MAP[smoking],
                bmi,
                hba1c,
                glucose,
            ]],
            columns=FEATURES,
        )
        pred = int(model.predict(row)[0])
        proba = float(model.predict_proba(row)[0][1]) * 100

        cls, icon, title = ("high", "🚨", "High risk of diabetes") if pred == 1 else ("low", "✅", "Low risk of diabetes")
        st.markdown(
            f'<div class="panel {cls}"><div class="icon">{icon}</div><h2>{title}</h2>'
            f'<p>Risk score: {proba:.0f}%</p>'
            f'<div class="bar"><div style="width:{max(proba, 3):.0f}%"></div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="panel empty"><div class="icon">📋</div><h2>Your result</h2>'
            "<p>Fill in the form and click Predict.</p></div>",
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="disclaimer"><b>Disclaimer:</b> This tool is for educational purposes only '
    "and is not a medical diagnosis. Please consult a doctor.</div>",
    unsafe_allow_html=True,
)
