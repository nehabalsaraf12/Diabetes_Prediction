"""
Diabetes Risk Predictor – Streamlit app
Model: KNeighborsClassifier (scikit-learn 1.6.1) loaded from diabetes_prediction.pkl
Features (in order): age, hypertension, heart_disease, smoking_history,
                     bmi, HbA1c_level, blood_glucose_level
"""

import pickle
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "diabetes_prediction.pkl"

# smoking_history was label-encoded alphabetically when the model was trained
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

# ----------------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Poppins', sans-serif;
}
#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background: linear-gradient(160deg, #eaf4fb 0%, #f4f8fb 45%, #e9f7f3 100%);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 860px;
}

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(120deg, #0f766e 0%, #0e7490 55%, #1d4ed8 100%);
    border-radius: 22px;
    padding: 2.2rem 2rem;
    text-align: center;
    color: #ffffff;
    box-shadow: 0 14px 34px rgba(15, 118, 110, 0.28);
    margin-bottom: 1.6rem;
}
.hero h1 {
    margin: 0;
    font-size: 2.1rem;
    font-weight: 700;
    color: #ffffff;
}
.hero p {
    margin: .55rem 0 0 0;
    font-size: 1rem;
    opacity: .92;
    color: #ffffff;
}

/* ---------- Section titles ---------- */
.section-title {
    display: flex;
    align-items: center;
    gap: .55rem;
    font-size: 1.1rem;
    font-weight: 600;
    color: #0f3d56;
    margin: .2rem 0 .8rem 0;
    padding-bottom: .45rem;
    border-bottom: 3px solid #14b8a6;
    width: fit-content;
}

/* ---------- Form card ---------- */
[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #dbe7f0;
    border-radius: 20px;
    padding: 1.6rem 1.6rem 1.2rem 1.6rem;
    box-shadow: 0 8px 26px rgba(18, 38, 58, 0.08);
}
label, .stMarkdown p, .stSelectbox label, .stNumberInput label, .stSlider label {
    color: #12263a !important;
    font-weight: 500 !important;
}

/* Submit button */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    width: 100%;
    background: linear-gradient(90deg, #0f766e, #0e7490);
    color: #ffffff;
    font-weight: 600;
    font-size: 1.05rem;
    border: none;
    border-radius: 14px;
    padding: .75rem 1rem;
    transition: transform .15s ease, box-shadow .15s ease;
    box-shadow: 0 8px 18px rgba(14, 116, 144, .28);
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 24px rgba(14, 116, 144, .38);
    color: #ffffff;
    border: none;
}

/* ---------- Result cards ---------- */
.result {
    border-radius: 20px;
    padding: 1.8rem 1.6rem;
    text-align: center;
    margin-top: 1.6rem;
    color: #ffffff;
    box-shadow: 0 12px 30px rgba(0,0,0,.14);
}
.result.low  { background: linear-gradient(135deg, #059669, #10b981); }
.result.mid  { background: linear-gradient(135deg, #d97706, #f59e0b); }
.result.high { background: linear-gradient(135deg, #be123c, #e11d48); }
.result .emoji { font-size: 2.6rem; }
.result h2 { margin: .3rem 0 .2rem 0; font-size: 1.6rem; color: #ffffff; }
.result p  { margin: 0; font-size: .98rem; opacity: .95; color: #ffffff; }

.meter-wrap {
    background: #ffffff;
    border: 1px solid #dbe7f0;
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
    box-shadow: 0 6px 20px rgba(18, 38, 58, 0.06);
}
.meter-label {
    display: flex;
    justify-content: space-between;
    font-size: .9rem;
    font-weight: 600;
    color: #12263a;
    margin-bottom: .5rem;
}
.meter {
    height: 14px;
    background: #e5edf3;
    border-radius: 999px;
    overflow: hidden;
}
.meter > div {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #10b981, #f59e0b, #e11d48);
    background-size: 600px 100%;
}

/* ---------- Summary chips ---------- */
.chips { display: flex; flex-wrap: wrap; gap: .6rem; margin-top: 1rem; }
.chip {
    background: #ffffff;
    border: 1px solid #dbe7f0;
    border-left: 5px solid #14b8a6;
    border-radius: 12px;
    padding: .55rem .9rem;
    font-size: .88rem;
    color: #12263a;
    box-shadow: 0 3px 10px rgba(18, 38, 58, .05);
}
.chip b { color: #0f766e; }
.chip.warn { border-left-color: #f59e0b; }
.chip.bad  { border-left-color: #e11d48; }

.disclaimer {
    margin-top: 1.6rem;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #7c2d12;
    border-radius: 14px;
    padding: .9rem 1.1rem;
    font-size: .85rem;
}
.footer {
    text-align: center;
    color: #5b7083;
    font-size: .8rem;
    margin-top: 1.4rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def html(block: str) -> None:
    """Render an HTML block (strips indentation so Markdown doesn't treat it as code)."""
    cleaned = "\n".join(line.strip() for line in block.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def bmi_category(bmi: float):
    if bmi < 18.5:
        return "Underweight", "warn"
    if bmi < 25:
        return "Normal", ""
    if bmi < 30:
        return "Overweight", "warn"
    return "Obese", "bad"


def hba1c_category(v: float):
    if v < 5.7:
        return "Normal range", ""
    if v < 6.5:
        return "Prediabetes range", "warn"
    return "Diabetes range", "bad"


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
html(
    """
    <div class="hero">
        <h1>🩺 Diabetes Risk Predictor</h1>
        <p>Enter a few health details and get an instant machine-learning estimate.</p>
    </div>
    """
)

try:
    model = load_model()
except Exception as e:  # noqa: BLE001
    st.error(
        "Could not load `diabetes_prediction.pkl`. Make sure it is in the same "
        "folder as app.py and that scikit-learn==1.6.1 is installed.\n\n"
        f"Details: {e}"
    )
    st.stop()

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
with st.form("patient_form"):
    html('<div class="section-title">📋 Patient details</div>')

    c1, c2 = st.columns(2, gap="large")

    with c1:
        age = st.number_input(
            "Age (years)", min_value=1.0, max_value=100.0, value=40.0, step=1.0
        )
        bmi = st.number_input(
            "BMI (kg/m²)", min_value=10.0, max_value=95.0, value=25.0, step=0.1,
            help="Body Mass Index. Training data range: 10 – 95.",
        )
        smoking = st.selectbox(
            "Smoking history", list(SMOKING_MAP.keys()), index=4
        )

    with c2:
        hba1c = st.number_input(
            "HbA1c level (%)", min_value=3.5, max_value=9.0, value=5.5, step=0.1,
            help="Average blood sugar over ~3 months. Training data range: 3.5 – 9.0.",
        )
        glucose = st.number_input(
            "Blood glucose level (mg/dL)", min_value=80, max_value=300, value=120, step=1,
            help="Training data range: 80 – 300 mg/dL.",
        )
        st.write("")
        hypertension = st.radio(
            "Hypertension", ["No", "Yes"], horizontal=True
        )
        heart_disease = st.radio(
            "Heart disease", ["No", "Yes"], horizontal=True
        )

    st.write("")
    submitted = st.form_submit_button("🔍 Predict diabetes risk")

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
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
    proba = float(model.predict_proba(row)[0][1])
    votes = round(proba * model.n_neighbors)
    pct = proba * 100

    if proba == 0:
        cls, emoji, title = "low", "✅", "Low risk"
        msg = "None of the most similar profiles in the training data had diabetes."
    elif pred == 0:
        cls, emoji, title = "mid", "⚠️", "Some risk indicators"
        msg = "The model predicts no diabetes, but some similar profiles did have it."
    else:
        cls, emoji, title = "high", "🚨", "High risk of diabetes"
        msg = "The majority of the most similar profiles in the training data had diabetes."

    html(
        f"""
        <div class="result {cls}">
            <div class="emoji">{emoji}</div>
            <h2>{title}</h2>
            <p>{msg}</p>
        </div>
        <div class="meter-wrap">
            <div class="meter-label">
                <span>Model risk score</span>
                <span>{pct:.0f}%</span>
            </div>
            <div class="meter"><div style="width:{max(pct, 3):.0f}%"></div></div>
            <div style="font-size:.82rem;color:#5b7083;margin-top:.6rem;">
                {votes} of the {model.n_neighbors} most similar patients in the training data had diabetes.
            </div>
        </div>
        """
    )

    bmi_txt, bmi_cls = bmi_category(bmi)
    a1c_txt, a1c_cls = hba1c_category(hba1c)
    html(
        f"""
        <div class="section-title" style="margin-top:1.4rem;">📊 Your inputs at a glance</div>
        <div class="chips">
            <div class="chip {bmi_cls}">BMI <b>{bmi:.1f}</b> · {bmi_txt}</div>
            <div class="chip {a1c_cls}">HbA1c <b>{hba1c:.1f}%</b> · {a1c_txt}</div>
            <div class="chip">Glucose <b>{glucose} mg/dL</b></div>
            <div class="chip">Age <b>{age:.0f}</b></div>
            <div class="chip {'bad' if hypertension == 'Yes' else ''}">Hypertension <b>{hypertension}</b></div>
            <div class="chip {'bad' if heart_disease == 'Yes' else ''}">Heart disease <b>{heart_disease}</b></div>
            <div class="chip">Smoking <b>{smoking}</b></div>
        </div>
        """
    )

html(
    """
    <div class="disclaimer">
        <b>⚕️ Disclaimer:</b> This tool is for educational purposes only and is not a
        medical diagnosis. Please consult a qualified healthcare professional for
        medical advice, testing and treatment.
    </div>
    <div class="footer">Built with Streamlit · scikit-learn K-Nearest Neighbors model</div>
    """
)
