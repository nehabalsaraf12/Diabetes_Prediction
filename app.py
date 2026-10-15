import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

/* page background */
.stApp {
    background: linear-gradient(160deg, #eef5ff 0%, #f6faff 45%, #eafaf5 100%);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

/* hero banner */
.hero {
    background: linear-gradient(120deg, #0f4c81 0%, #1c7ed6 55%, #12b886 100%);
    border-radius: 22px;
    padding: 2.2rem 2.4rem;
    color: #ffffff;
    box-shadow: 0 12px 30px rgba(28, 126, 214, 0.28);
    margin-bottom: 1.6rem;
}
.hero h1 { margin: 0; font-size: 2.2rem; font-weight: 700; color: #ffffff; }
.hero p  { margin: .5rem 0 0 0; font-size: 1.02rem; opacity: .93; color: #ffffff; }

/* cards */
.card {
    background: #ffffff;
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 6px 22px rgba(15, 76, 129, 0.08);
    border: 1px solid #e3edf9;
    margin-bottom: 1.2rem;
}
.card h3 { margin: 0 0 .2rem 0; color: #0f4c81; font-size: 1.15rem; font-weight: 600; }
.card .sub { color: #6b7c93; font-size: .88rem; margin-bottom: .4rem; }

/* form container */
div[data-testid="stForm"] {
    background: #ffffff;
    border-radius: 18px;
    border: 1px solid #e3edf9;
    box-shadow: 0 6px 22px rgba(15, 76, 129, 0.08);
    padding: 1.4rem 1.6rem;
}
div[data-testid="stForm"] label, div[data-testid="stForm"] p {
    color: #243b53 !important;
    font-weight: 500;
}

/* submit button */
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    background: linear-gradient(90deg, #1c7ed6, #12b886);
    color: #ffffff;
    border: none;
    border-radius: 12px;
    padding: .7rem 1rem;
    font-size: 1.05rem;
    font-weight: 600;
    transition: transform .15s ease, box-shadow .15s ease;
}
div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(28, 126, 214, 0.35);
    color: #ffffff;
    border: none;
}

/* result cards */
.result {
    border-radius: 20px;
    padding: 1.8rem;
    text-align: center;
    color: #ffffff;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.15);
    margin-bottom: 1.2rem;
}
.result.high { background: linear-gradient(135deg, #e03131, #ff6b6b); }
.result.mid  { background: linear-gradient(135deg, #f08c00, #ffa94d); }
.result.low  { background: linear-gradient(135deg, #0ca678, #38d9a9); }
.result .icon  { font-size: 3rem; line-height: 1; }
.result .title { font-size: 1.5rem; font-weight: 700; margin-top: .4rem; color: #ffffff; }
.result .desc  { font-size: .95rem; opacity: .95; margin-top: .3rem; color: #ffffff; }

/* vote bar */
.vote-wrap { margin-top: 1.1rem; text-align: left; }
.vote-label { font-size: .85rem; color: #ffffff; margin-bottom: .3rem; opacity: .95; }
.vote-bar { background: rgba(255,255,255,.3); border-radius: 999px; height: 12px; overflow: hidden; }
.vote-fill { background: #ffffff; height: 100%; border-radius: 999px; }

/* metric chips */
.chips { display: flex; flex-direction: column; gap: .6rem; margin-top: .6rem; }
.chip {
    display: flex; justify-content: space-between; align-items: center;
    background: #f5f9ff; border: 1px solid #e3edf9;
    border-radius: 12px; padding: .65rem .9rem;
}
.chip .k { color: #486581; font-size: .88rem; font-weight: 500; }
.chip .v { font-weight: 600; font-size: .92rem; }
.v.good { color: #0ca678; }
.v.warn { color: #f08c00; }
.v.bad  { color: #e03131; }

/* placeholder */
.placeholder { text-align: center; padding: 2.2rem 1rem; color: #6b7c93; }
.placeholder .big { font-size: 3rem; }

/* disclaimer */
.disclaimer {
    background: #fff9db; border-left: 5px solid #fab005;
    color: #5c4a00; border-radius: 10px;
    padding: .9rem 1.1rem; font-size: .88rem; margin-top: .6rem;
}
.footer { text-align: center; color: #829ab1; font-size: .82rem; margin-top: 1.6rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------
# Load models (cached)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

# Encoding used while training (alphabetical label encoding of smoking_history)
SMOKING_MAP = {
    "No Info": 0,
    "Current": 1,
    "Ever": 2,
    "Former": 3,
    "Never": 4,
    "Not Current": 5,
}

IMPROVED_K = 15


@st.cache_resource(show_spinner="Loading model (first run only)...")
def load_models():
    """Returns (original_model, improved_model).

    The original KNN was trained on UNSCALED data, so blood glucose (range 80-300)
    dominates the distance and the other inputs barely matter. The pickle stores its
    training data, so we rebuild the same KNN with StandardScaler for better results.
    """
    model_path = BASE_DIR / "diabetes.pkl"
    if not model_path.exists():
        st.error(
            "`diabetes.pkl` was not found next to app.py. "
            f"Files here: {sorted(p.name for p in BASE_DIR.iterdir())}"
        )
        st.stop()

    with open(model_path, "rb") as f:
        original = pickle.load(f)

    X = pd.DataFrame(np.asarray(original._fit_X), columns=FEATURES)
    y = np.asarray(original._y)
    improved = make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=IMPROVED_K)
    ).fit(X, y)
    return original, improved


original_model, improved_model = load_models()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def bmi_status(bmi):
    if bmi < 18.5:
        return "Underweight", "warn"
    if bmi < 25:
        return "Normal", "good"
    if bmi < 30:
        return "Overweight", "warn"
    return "Obese", "bad"


def hba1c_status(v):
    if v < 5.7:
        return "Normal", "good"
    if v < 6.5:
        return "Prediabetes range", "warn"
    return "Diabetes range", "bad"


def glucose_status(v):
    if v < 140:
        return "Normal", "good"
    if v < 200:
        return "Elevated", "warn"
    return "High", "bad"


# ------------------------------------------------------------------
# Hero header
# ------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
    <h1>🩺 Diabetes Risk Predictor</h1>
    <p>Enter a few health details and our machine-learning model (K-Nearest Neighbors)
    will estimate whether the profile is similar to patients with diabetes.</p>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 1], gap="large")

# ------------------------------------------------------------------
# Left column: input form
# ------------------------------------------------------------------
with left:
    st.markdown(
        '<div class="card"><h3>📝 Patient Information</h3>'
        '<div class="sub">Fill in the details below and press Predict.</div></div>',
        unsafe_allow_html=True,
    )
    with st.form("patient_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age (years)", 1, 100, 40)
            bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
            hba1c = st.number_input(
                "HbA1c level (%)", min_value=3.5, max_value=9.0, value=5.5, step=0.1
            )
            glucose = st.number_input(
                "Blood glucose level (mg/dL)", min_value=80, max_value=300, value=120, step=1
            )
        with c2:
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart disease", ["No", "Yes"])
            smoking = st.selectbox("Smoking history", list(SMOKING_MAP.keys()), index=0)
            model_choice = st.selectbox(
                "Model",
                ["Improved (scaled KNN)", "Original (your .pkl)"],
                help="Improved = same training data with feature scaling, so every input matters.",
            )

        submitted = st.form_submit_button("🔍 Predict Diabetes Risk")

# ------------------------------------------------------------------
# Right column: results
# ------------------------------------------------------------------
with right:
    if not submitted:
        st.markdown(
            """
<div class="card placeholder">
    <div class="big">📊</div>
    <h3>Your result will appear here</h3>
    <div class="sub">Complete the form on the left and click <b>Predict Diabetes Risk</b>.</div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        input_df = pd.DataFrame(
            [
                [
                    age,
                    1 if hypertension == "Yes" else 0,
                    1 if heart_disease == "Yes" else 0,
                    SMOKING_MAP[smoking],
                    bmi,
                    hba1c,
                    glucose,
                ]
            ],
            columns=FEATURES,
        )

        use_improved = model_choice.startswith("Improved")
        model = improved_model if use_improved else original_model
        k = IMPROVED_K if use_improved else original_model.n_neighbors

        p_diab = float(model.predict_proba(input_df)[0][1])
        votes = int(round(p_diab * k))
        pct = int(round(p_diab * 100))

        if p_diab >= 0.5:
            css, icon, title = "high", "⚠️", "High Risk of Diabetes"
            desc = "This profile closely resembles patients who have diabetes."
        elif p_diab >= 0.2:
            css, icon, title = "mid", "🟠", "Moderate Risk of Diabetes"
            desc = "This profile partly resembles patients with diabetes. Consider a check-up."
        else:
            css, icon, title = "low", "✅", "Low Risk of Diabetes"
            desc = "This profile mostly resembles patients without diabetes."

        st.markdown(
            f"""
<div class="result {css}">
    <div class="icon">{icon}</div>
    <div class="title">{title}</div>
    <div class="desc">{desc}</div>
    <div class="vote-wrap">
        <div class="vote-label">Risk score: {pct}% &nbsp;|&nbsp; {votes} of {k} most similar patients had diabetes</div>
        <div class="vote-bar"><div class="vote-fill" style="width:{max(pct, 4)}%"></div></div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        b_txt, b_cls = bmi_status(bmi)
        h_txt, h_cls = hba1c_status(hba1c)
        g_txt, g_cls = glucose_status(glucose)

        st.markdown(
            f"""
<div class="card">
    <h3>📋 Health Indicators</h3>
    <div class="chips">
        <div class="chip"><span class="k">BMI ({bmi:.1f})</span><span class="v {b_cls}">{b_txt}</span></div>
        <div class="chip"><span class="k">HbA1c ({hba1c:.1f}%)</span><span class="v {h_cls}">{h_txt}</span></div>
        <div class="chip"><span class="k">Blood glucose ({glucose} mg/dL)</span><span class="v {g_cls}">{g_txt}</span></div>
        <div class="chip"><span class="k">Hypertension</span><span class="v {'bad' if hypertension == 'Yes' else 'good'}">{hypertension}</span></div>
        <div class="chip"><span class="k">Heart disease</span><span class="v {'bad' if heart_disease == 'Yes' else 'good'}">{heart_disease}</span></div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        with st.expander("View input data sent to the model"):
            st.dataframe(input_df, hide_index=True)

# ------------------------------------------------------------------
# Disclaimer & footer
# ------------------------------------------------------------------
st.markdown(
    """
<div class="disclaimer">
    <b>Disclaimer:</b> This tool is for educational purposes only and is not a medical diagnosis.
    Please consult a qualified healthcare professional for medical advice.
</div>
<div class="footer">Built with Streamlit • KNN Classifier</div>
""",
    unsafe_allow_html=True,
)
