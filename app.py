
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "diabetes.pkl"

FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

# Assumed alphabetical LabelEncoder mapping.
# Verify this against your original training code.
SMOKING_MAP = {
    "No information": 0,
    "Current": 1,
    "Ever": 2,
    "Former": 3,
    "Never": 4,
    "Not current": 5
}

# ---------------- CUSTOM HTML AND CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #eef7fa, #f7f9fc);
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(120deg, #10233f, #087f8c);
    color: white;
    padding: 38px;
    border-radius: 22px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(16,35,63,0.15);
}

.hero h1 {
    color: white;
    font-size: 38px;
    font-weight: 700;
}

.hero p {
    color: #e3f5f7;
    font-size: 16px;
    line-height: 1.7;
}

.section-title {
    color: #10233f;
    font-size: 24px;
    font-weight: 700;
    margin: 20px 0 5px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border: 1px solid #e0e8ef;
    border-radius: 16px;
    padding: 20px;
}

div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    color: #243b53;
    font-weight: 600;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(100deg, #087f8c, #10a4a0);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: 700;
    min-height: 48px;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    color: white;
    border: none;
    background: #076b76;
}

.result {
    padding: 24px;
    border-radius: 15px;
    margin-top: 20px;
    line-height: 1.7;
}

.positive {
    background: #fff4e5;
    border-left: 6px solid #e39a24;
}

.negative {
    background: #e9f8f3;
    border-left: 6px solid #07866f;
}

.warning {
    background: #fff8e8;
    color: #74531b;
    padding: 14px;
    border-radius: 10px;
    margin-top: 20px;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #758398;
    padding: 30px 0 10px;
    font-size: 13px;
}
</style>

<div class="hero">
    <h1>🩺 Diabetes Prediction System</h1>
    <p>
        Machine-learning-based health screening demonstration.
        Enter your health measurements to view your model's prediction.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "diabetes.pkl must be in the same folder as app.py."
        )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    model_features = list(
        getattr(model, "feature_names_in_", FEATURES)
    )

    if model_features != FEATURES:
        raise ValueError(
            f"Feature order mismatch. Model expects: {model_features}"
        )

    return model


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


# ---------------- INPUT FORM ----------------
st.markdown(
    '<div class="section-title">Enter Patient Details</div>',
    unsafe_allow_html=True
)

st.write("Fill in all seven fields below.")

with st.form("diabetes_form"):

    st.subheader("Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=100,
            value=35
        )

    with col2:
        hypertension = st.selectbox(
            "Hypertension",
            ["No", "Yes"]
        )

    with col3:
        heart_disease = st.selectbox(
            "Heart Disease",
            ["No", "Yes"]
        )

    st.subheader("Health Measurements")

    col4, col5 = st.columns(2)

    with col4:
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=10.0,
            max_value=80.0,
            value=25.0,
            step=0.1
        )

        hba1c = st.number_input(
            "HbA1c Level (%)",
            min_value=3.0,
            max_value=20.0,
            value=5.7,
            step=0.1
        )

    with col5:
        glucose = st.number_input(
            "Blood Glucose (mg/dL)",
            min_value=50,
            max_value=500,
            value=100
        )

        smoking = st.selectbox(
            "Smoking History",
            list(SMOKING_MAP.keys())
        )

    submitted = st.form_submit_button(
        "🔍 Predict Diabetes",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------
if submitted:

    # Age scaling assumes training used age / 100.
    input_data = pd.DataFrame([{
        "age": age / 100.0,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "smoking_history": SMOKING_MAP[smoking],
        "bmi": float(bmi),
        "HbA1c_level": float(hba1c),
        "blood_glucose_level": float(glucose)
    }], columns=FEATURES)

    try:
        prediction = float(
            np.asarray(model.predict(input_data)).reshape(-1)[0]
        )

        st.markdown("---")
        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)

        m1.metric("Age", f"{age} years")
        m2.metric("BMI", f"{bmi:.1f}")
        m3.metric("Blood Glucose", f"{glucose} mg/dL")

        # This assumes the regressor was trained on 0/1 targets.
        if prediction >= 0.5:
            st.markdown("""
            <div class="result positive">
                <h3>⚠️ Model Output: Positive</h3>
                The model output is at or above the 0.5 threshold.
                This does not confirm that you have diabetes.
                Consult a qualified healthcare professional.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="result negative">
                <h3>✓ Model Output: Negative</h3>
                The model output is below the 0.5 threshold.
                This does not rule out diabetes. Seek medical advice
                if you have symptoms or concerns.
            </div>
            """, unsafe_allow_html=True)

        st.caption(
            f"Raw model output: {prediction:.4f}. "
            "This is not a calibrated probability."
        )

        with st.expander("View model input values"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.markdown("""
<div class="warning">
    <strong>Medical Disclaimer:</strong>
    This application is an educational machine-learning demonstration,
    not a medical diagnostic tool. Model results can be incorrect.
    Always consult a healthcare professional for medical decisions.
</div>

<div class="footer">
    Diabetes Prediction System | Built with Python and Streamlit
</div>
""", unsafe_allow_html=True)
