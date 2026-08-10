from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
  page_title="CardioAI | Heart Health Dashboard",
  page_icon="🫀",
  layout="wide",
  initial_sidebar_state="expanded",
)


def inject_styles():
  st.markdown(
    """
    <style>
    :root {
      --bg-1: #eaf4ff;
      --bg-2: #f7f1ff;
      --bg-3: #edfdf6;
      --ink: #0f2747;
      --muted: #58708f;
      --card: rgba(255, 255, 255, 0.72);
      --border: rgba(129, 156, 189, 0.25);
      --shadow: 0 18px 60px rgba(15, 39, 71, 0.10);
      --accent: #ff5b6e;
      --accent-2: #f47b7b;
      --cyan: #69d7e8;
      --lavender: #b9a7ff;
      --green: #52c1a3;
    }

    .stApp {
      background:
        radial-gradient(circle at 12% 18%, rgba(185, 167, 255, 0.25), transparent 28%),
        radial-gradient(circle at 88% 12%, rgba(105, 215, 232, 0.22), transparent 24%),
        radial-gradient(circle at 78% 88%, rgba(82, 193, 163, 0.16), transparent 25%),
        linear-gradient(135deg, var(--bg-1), var(--bg-2) 48%, var(--bg-3));
      color: var(--ink);
    }

    .main .block-container {
      padding-top: 1.1rem;
      padding-bottom: 2.3rem;
      max-width: 1320px;
    }

    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.82), rgba(239, 246, 255, 0.72));
      backdrop-filter: blur(18px);
      border-right: 1px solid rgba(129, 156, 189, 0.16);
    }

    [data-testid="stSidebar"] .block-container {
      padding-top: 1.4rem;
    }

    .premium-shell {
      background: rgba(255, 255, 255, 0.20);
      border: 1px solid rgba(255, 255, 255, 0.28);
      border-radius: 26px;
      box-shadow: var(--shadow);
      padding: 1.15rem;
      backdrop-filter: blur(22px);
    }

    .hero-card {
      background: linear-gradient(135deg, rgba(255, 255, 255, 0.88), rgba(255, 255, 255, 0.64));
      border: 1px solid rgba(129, 156, 189, 0.22);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 1.3rem 1.35rem;
      backdrop-filter: blur(20px);
    }

    .hero-title {
      font-size: 2.55rem;
      line-height: 1.04;
      font-weight: 800;
      color: var(--ink);
      margin: 0;
      letter-spacing: -0.03em;
    }

    .hero-subtitle {
      color: var(--accent);
      font-size: 1.02rem;
      font-weight: 700;
      margin-top: 0.35rem;
    }

    .hero-copy {
      color: var(--muted);
      font-size: 1rem;
      margin-top: 0.65rem;
      max-width: 760px;
    }

    .badge-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem;
      margin-top: 0.9rem;
    }

    .soft-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.42rem;
      padding: 0.52rem 0.78rem;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.68);
      border: 1px solid rgba(129, 156, 189, 0.18);
      color: var(--ink);
      font-size: 0.88rem;
      font-weight: 600;
      box-shadow: 0 8px 24px rgba(15, 39, 71, 0.05);
    }

    .glass-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 1rem 1.05rem;
      backdrop-filter: blur(18px);
    }

    .glass-card.dark {
      background: linear-gradient(180deg, rgba(9, 28, 57, 0.92), rgba(15, 39, 71, 0.82));
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #eef5ff;
    }

    .metric-eyebrow {
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: rgba(15, 39, 71, 0.66);
      font-weight: 700;
      margin-bottom: 0.3rem;
    }

    .metric-title {
      font-size: 1.55rem;
      font-weight: 800;
      line-height: 1.05;
      color: var(--ink);
    }

    .metric-title.dark {
      color: #ffffff;
    }

    .metric-sub {
      margin-top: 0.35rem;
      font-size: 0.9rem;
      color: var(--muted);
    }

    .metric-sub.dark {
      color: rgba(238, 245, 255, 0.8);
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0.9rem;
      margin-top: 0.9rem;
    }

    .summary-card {
      min-height: 112px;
      border-radius: 24px;
      padding: 1rem 1rem 0.95rem 1rem;
      background: linear-gradient(135deg, rgba(255, 255, 255, 0.78), rgba(255, 255, 255, 0.56));
      border: 1px solid rgba(129, 156, 189, 0.18);
      box-shadow: 0 14px 36px rgba(15, 39, 71, 0.07);
    }

    .summary-card .label {
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }

    .summary-card .value {
      margin-top: 0.42rem;
      color: var(--ink);
      font-size: 1.34rem;
      font-weight: 800;
      letter-spacing: -0.02em;
    }

    .summary-card .hint {
      margin-top: 0.28rem;
      color: rgba(15, 39, 71, 0.68);
      font-size: 0.88rem;
    }

    @media (max-width: 1080px) {
      .summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    .section-shell {
      margin-top: 1.05rem;
      background: rgba(255, 255, 255, 0.58);
      border: 1px solid rgba(129, 156, 189, 0.18);
      border-radius: 26px;
      box-shadow: 0 18px 48px rgba(15, 39, 71, 0.07);
      padding: 1rem 1rem 0.85rem 1rem;
      backdrop-filter: blur(18px);
    }

    .section-title {
      display: flex;
      align-items: center;
      gap: 0.7rem;
      font-size: 1.08rem;
      font-weight: 800;
      color: var(--ink);
      margin-bottom: 0.2rem;
    }

    .section-desc {
      color: var(--muted);
      font-size: 0.93rem;
      margin-bottom: 0.9rem;
    }

    .prediction-btn button {
      width: 100%;
      height: 3.45rem;
      border-radius: 18px !important;
      font-size: 1.02rem !important;
      font-weight: 800 !important;
      letter-spacing: 0.02em;
      background: linear-gradient(135deg, #ff5b6e, #ff6f8d, #f47b7b) !important;
      color: white !important;
      border: none !important;
      box-shadow: 0 16px 34px rgba(255, 91, 110, 0.28);
    }

    .prediction-btn button:hover {
      transform: translateY(-1px);
      box-shadow: 0 18px 38px rgba(255, 91, 110, 0.32);
    }

    .risk-card {
      background: linear-gradient(135deg, rgba(255, 255, 255, 0.90), rgba(255, 255, 255, 0.66));
      border: 1px solid rgba(129, 156, 189, 0.20);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 1.15rem 1.2rem;
      backdrop-filter: blur(18px);
    }

    .risk-title {
      font-size: 1.28rem;
      font-weight: 800;
      color: var(--ink);
      margin-bottom: 0.4rem;
    }

    .risk-sub {
      color: var(--muted);
      font-size: 0.92rem;
      margin-bottom: 0.9rem;
    }

    .risk-value {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 1rem;
      margin-bottom: 0.8rem;
    }

    .risk-value .score {
      font-size: 2.05rem;
      font-weight: 900;
      color: var(--ink);
      letter-spacing: -0.04em;
    }

    .risk-value .status {
      font-size: 0.95rem;
      font-weight: 800;
      color: var(--accent);
      text-align: right;
    }

    .risk-track {
      width: 100%;
      height: 16px;
      background: rgba(15, 39, 71, 0.08);
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid rgba(129, 156, 189, 0.15);
    }

    .risk-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.35s ease;
    }

    .risk-fill.low {
      background: linear-gradient(90deg, #66c7ac, #69d7e8);
    }

    .risk-fill.high {
      background: linear-gradient(90deg, #ff7b86, #ff5b6e);
    }

    .result-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      padding: 0.45rem 0.8rem;
      border-radius: 999px;
      font-weight: 800;
      font-size: 0.86rem;
      margin-top: 0.75rem;
    }

    .result-pill.low {
      color: #15795f;
      background: rgba(82, 193, 163, 0.14);
      border: 1px solid rgba(82, 193, 163, 0.20);
    }

    .result-pill.high {
      color: #a11f32;
      background: rgba(255, 91, 110, 0.14);
      border: 1px solid rgba(255, 91, 110, 0.20);
    }

    .model-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.8rem;
      margin-top: 0.9rem;
    }

    .model-card {
      background: linear-gradient(135deg, rgba(255, 255, 255, 0.82), rgba(255, 255, 255, 0.60));
      border: 1px solid rgba(129, 156, 189, 0.18);
      border-radius: 22px;
      padding: 0.9rem 0.95rem;
    }

    .model-card .kicker {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
      font-weight: 700;
    }

    .model-card .metric {
      margin-top: 0.3rem;
      font-size: 1.3rem;
      font-weight: 900;
      color: var(--ink);
    }

    .disclaimer {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.55;
    }

    .stNumberInput input, .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
      border-radius: 16px !important;
    }

    .stNumberInput label, .stSelectbox label {
      color: var(--ink) !important;
      font-weight: 650 !important;
    }

    @media (max-width: 720px) {
      .hero-title {
        font-size: 1.9rem;
      }

      .model-grid {
        grid-template-columns: 1fr;
      }

      .summary-grid {
        grid-template-columns: 1fr;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
  )


inject_styles()


@st.cache_resource
def load_pipeline():
  return joblib.load("heart_disease_xgb_pipeline.pkl")


try:
  pipeline = load_pipeline()
except Exception:
  st.error(
    "Unable to load 'heart_disease_xgb_pipeline.pkl'. The Streamlit app can"
    " not continue without the existing trained pipeline."
  )
  st.stop()


def engineer_medical_features(df):
  df_fe = df.copy()
  df_fe["Cholesterol_HDL_Ratio"] = df_fe["Total_Cholesterol"] / (
    df_fe["HDL"] + 1e-5
  )
  df_fe["LDL_HDL_Ratio"] = df_fe["LDL"] / (df_fe["HDL"] + 1e-5)
  df_fe["Triglyceride_HDL_Ratio"] = df_fe["Triglycerides"] / (
    df_fe["HDL"] + 1e-5
  )
  df_fe["AIP"] = np.log10(
    (df_fe["Triglycerides"] / (df_fe["HDL"] + 1e-5)) + 1e-5
  )
  df_fe["Pulse_Pressure"] = df_fe["Resting_BP"] - df_fe["Diastolic_BP"]
  df_fe["MAP"] = (
    df_fe["Diastolic_BP"]
    + (df_fe["Resting_BP"] - df_fe["Diastolic_BP"]) / 3.0
  )
  df_fe["Heart_Rate_Reserve"] = (
    df_fe["Maximum_Heart_Rate"] - df_fe["Resting_Heart_Rate"]
  )
  df_fe["Waist_to_BMI_Ratio"] = df_fe["Waist_Circumference"] / (
    df_fe["BMI"] + 1e-5
  )
  return df_fe


def section_header(icon, title, subtitle):
  st.markdown(
    f"""
    <div class="section-title">{icon} {title}</div>
    <div class="section-desc">{subtitle}</div>
    """,
    unsafe_allow_html=True,
  )


def summary_card(label, value, hint, accent_class=""):
  return f"""
  <div class="summary-card {accent_class}">
    <div class="label">{label}</div>
    <div class="value">{value}</div>
    <div class="hint">{hint}</div>
  </div>
  """


def sidebar_metric(label, value, note):
  return f"""
  <div class="model-card">
    <div class="kicker">{label}</div>
    <div class="metric">{value}</div>
    <div class="hint">{note}</div>
  </div>
  """


def load_optional_heart_image():
  candidates = [
    Path("assets") / "heart.png",
    Path(__file__).resolve().parent / "assets" / "heart.png",
  ]
  for candidate in candidates:
    if candidate.exists():
      return candidate
  return None


with st.sidebar:
  st.markdown(
    """
    <div class="glass-card dark">
      <div class="metric-eyebrow">CardioAI</div>
      <div class="metric-title dark">Premium Heart Health Dashboard</div>
      <div class="metric-sub dark">
        AI-powered cardiovascular risk assessment using the existing trained pipeline.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
  )
  st.markdown("<div style='height: 0.85rem;'></div>", unsafe_allow_html=True)
  st.markdown(
    sidebar_metric(
      "Model",
      "XGBoost",
      "The inference pipeline is loaded from the saved model artifact.",
    ),
    unsafe_allow_html=True,
  )
  st.markdown(
    sidebar_metric(
      "Decision Rule",
      "0.40",
      "Probabilities at or above this threshold are labeled elevated risk.",
    ),
    unsafe_allow_html=True,
  )
  st.markdown(
    sidebar_metric(
      "Experience",
      "Glass UI",
      "Soft gradients, rounded cards and responsive spacing for clinical clarity.",
    ),
    unsafe_allow_html=True,
  )
  st.markdown(
    """
    <div class="glass-card" style="margin-top: 0.85rem;">
      <div class="metric-eyebrow">Safety Note</div>
      <div class="disclaimer">
        This interface is for an educational machine-learning project and is not a medical diagnosis.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
  )


st.markdown(
  """
  <div class="premium-shell">
  """,
  unsafe_allow_html=True,
)

header_col1, header_col2 = st.columns([1.55, 0.7], vertical_alignment="center")

with header_col1:
  st.markdown(
    """
    <div class="hero-card">
      <div class="hero-title">🫀 CardioAI</div>
      <div class="hero-subtitle">AI-Powered Heart Health Assessment</div>
      <div class="hero-copy">
        Assess cardiovascular risk using clinical, lifestyle and medical health indicators.
        The output is a model-based screening estimate, not a medical diagnosis.
      </div>
      <div class="badge-row">
        <span class="soft-badge">Clinical inputs preserved</span>
        <span class="soft-badge">XGBoost pipeline intact</span>
        <span class="soft-badge">Educational project</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
  )

with header_col2:
  heart_image = load_optional_heart_image()
  if heart_image is not None:
    st.markdown(
      """
      <div class="hero-card" style="display:flex; align-items:center; justify-content:center; min-height: 198px;">
      """,
      unsafe_allow_html=True,
    )
    st.image(str(heart_image), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
  else:
    st.markdown(
      """
      <div class="hero-card" style="min-height: 198px; display:flex; flex-direction:column; justify-content:center;">
        <div class="metric-eyebrow">Cardio View</div>
        <div class="metric-title">Healthy interface, same model.</div>
        <div class="metric-sub">
          Optional artwork can be placed in <strong>assets/heart.png</strong>.
        </div>
      </div>
      """,
      unsafe_allow_html=True,
    )

st.markdown(
  "<div style='height: 0.95rem;'></div>",
  unsafe_allow_html=True,
)

st.markdown('<div class="summary-grid">', unsafe_allow_html=True)
summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
  st.markdown(
    summary_card(
      "Heart Risk",
      "Pending" if "cardio_last_proba" not in st.session_state else f"{st.session_state['cardio_last_proba'] * 100:.2f}%",
      "Exact model probability appears after assessment.",
    ),
    unsafe_allow_html=True,
  )

with summary_col2:
  st.markdown(
    summary_card(
      "Blood Pressure",
      "{}/{} mmHg".format(
        int(st.session_state.get("cardio_resting_bp", 130)),
        int(st.session_state.get("cardio_diastolic_bp", 80)),
      ),
      "Current systolic and diastolic readings from the active inputs.",
    ),
    unsafe_allow_html=True,
  )

with summary_col3:
  st.markdown(
    summary_card(
      "Heart Rate",
      "{} / {} bpm".format(
        int(st.session_state.get("cardio_rest_hr", 72)),
        int(st.session_state.get("cardio_max_hr", 150)),
      ),
      "Resting and maximum heart rate are shown as entered.",
    ),
    unsafe_allow_html=True,
  )

with summary_col4:
  st.markdown(
    summary_card(
      "Health Profile",
      "BMI {} | Steps {:,}".format(
        f"{float(st.session_state.get('cardio_bmi', 26.5)):.1f}",
        int(st.session_state.get("cardio_steps", 7000)),
      ),
      "A compact snapshot of key lifestyle and anthropometric inputs.",
    ),
    unsafe_allow_html=True,
  )

st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-shell">', unsafe_allow_html=True)
section_header(
  "👤",
  "Personal & Vitals",
  "Core demographic and body-measurement inputs used as entered by the user.",
)
personal_col1, personal_col2, personal_col3 = st.columns(3)

with personal_col1:
  age = st.number_input("Age", 20, 100, 55, key="cardio_age")
  sex = st.selectbox("Sex", ["Male", "Female"], key="cardio_sex")

with personal_col2:
  bmi = st.number_input("BMI", 15.0, 50.0, 26.5, key="cardio_bmi")
  waist_circ = st.number_input(
    "Waist Circumference (cm)", 50, 150, 90, key="cardio_waist"
  )

with personal_col3:
  rest_hr = st.number_input(
    "Resting Heart Rate", 40, 120, 72, key="cardio_rest_hr"
  )
  max_hr = st.number_input(
    "Maximum Heart Rate", 60, 220, 150, key="cardio_max_hr"
  )
st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-shell">', unsafe_allow_html=True)
section_header(
  "🩸",
  "Blood & Lipid Profile",
  "Systolic and diastolic pressure, lipid panel, glycemic markers and related measurements.",
)
blood_col1, blood_col2, blood_col3, blood_col4 = st.columns(4)

with blood_col1:
  resting_bp = st.number_input(
    "Resting Systolic BP (mmHg)", 80, 200, 130, key="cardio_resting_bp"
  )
  diastolic_bp = st.number_input(
    "Diastolic BP (mmHg)", 50, 120, 80, key="cardio_diastolic_bp"
  )

with blood_col2:
  cholesterol = st.number_input(
    "Total Cholesterol (mg/dL)", 100, 500, 220, key="cardio_cholesterol"
  )
  ldl = st.number_input(
    "LDL Cholesterol (mg/dL)", 50, 300, 130, key="cardio_ldl"
  )

with blood_col3:
  hdl = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, 50, key="cardio_hdl")
  triglycerides = st.number_input(
    "Triglycerides (mg/dL)", 50, 500, 150, key="cardio_triglycerides"
  )

with blood_col4:
  fbs = st.number_input(
    "Fasting Blood Sugar (mg/dL)", 70, 250, 100, key="cardio_fbs"
  )
  hba1c = st.number_input("HbA1c (%)", 4.0, 12.0, 5.7, key="cardio_hba1c")
st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-shell">', unsafe_allow_html=True)
section_header(
  "❤️",
  "Cardiac Assessment",
  "Clinical cardiac indicators and electrocardiographic or exercise-related markers.",
)
cardiac_col1, cardiac_col2, cardiac_col3 = st.columns(3)

with cardiac_col1:
  chest_pain = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
    key="cardio_chest_pain",
  )
  ex_angina = st.selectbox(
    "Exercise Induced Angina", ["No", "Yes"], key="cardio_ex_angina"
  )
  ecg = st.selectbox(
    "Resting ECG", ["Normal", "ST-T Abnormality", "LVH"], key="cardio_ecg"
  )

with cardiac_col2:
  st_dep = st.number_input(
    "ST Depression", 0.0, 6.0, 1.0, key="cardio_st_dep"
  )
  st_slope = st.selectbox(
    "ST Slope", ["Upsloping", "Flat", "Downsloping"], key="cardio_st_slope"
  )

with cardiac_col3:
  major_vessels = st.selectbox(
    "Major Vessels (0-4)", [0, 1, 2, 3, 4], key="cardio_major_vessels"
  )
  thalassemia = st.selectbox(
    "Thalassemia",
    ["Normal", "Fixed Defect", "Reversible Defect"],
    key="cardio_thalassemia",
  )
st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-shell">', unsafe_allow_html=True)
section_header(
  "🏃",
  "Lifestyle",
  "Everyday behavior, sleep, stress and activity markers that remain editable inputs.",
)
lifestyle_col1, lifestyle_col2, lifestyle_col3 = st.columns(3)

with lifestyle_col1:
  smoking = st.selectbox(
    "Smoking Status", ["Never", "Former", "Current"], key="cardio_smoking"
  )
  alcohol = st.selectbox(
    "Alcohol Consumption", ["Never", "Occasional", "Frequent"], key="cardio_alcohol"
  )
  activity = st.selectbox(
    "Physical Activity", ["Low", "Moderate", "High"], key="cardio_activity"
  )

with lifestyle_col2:
  steps = st.number_input(
    "Daily Steps", 500, 20000, 7000, key="cardio_steps"
  )
  stress = st.selectbox(
    "Stress Level", ["Low", "Medium", "High"], key="cardio_stress"
  )

with lifestyle_col3:
  sleep = st.number_input(
    "Sleep Hours", 3.0, 12.0, 7.0, key="cardio_sleep"
  )
  diet = st.selectbox(
    "Diet Quality", ["Poor", "Average", "Good"], key="cardio_diet"
  )
st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-shell">', unsafe_allow_html=True)
section_header(
  "🧬",
  "Medical History",
  "Self-reported medical history indicators kept exactly as original inputs.",
)
history_col1, history_col2 = st.columns(2)

with history_col1:
  family_hist = st.selectbox(
    "Family History of Heart Disease", ["No", "Yes"], key="cardio_family_hist"
  )
  diabetes = st.selectbox("Diabetes", ["No", "Yes"], key="cardio_diabetes")

with history_col2:
  hypertension = st.selectbox(
    "Hypertension", ["No", "Yes"], key="cardio_hypertension"
  )
  ckd = st.selectbox(
    "Chronic Kidney Disease", ["No", "Yes"], key="cardio_ckd"
  )
st.markdown("</div>", unsafe_allow_html=True)


button_col1, button_col2, button_col3 = st.columns([1.2, 1.0, 1.2], vertical_alignment="center")
with button_col2:
  assess_clicked = st.button("❤️  ASSESS HEART HEALTH", type="primary", use_container_width=True)

input_data = {
  "Age": age,
  "Sex": sex,
  "Chest_Pain_Type": chest_pain,
  "Resting_BP": resting_bp,
  "Diastolic_BP": diastolic_bp,
  "Total_Cholesterol": cholesterol,
  "LDL": ldl,
  "HDL": hdl,
  "Triglycerides": triglycerides,
  "Fasting_Blood_Sugar": fbs,
  "HbA1c": hba1c,
  "BMI": bmi,
  "Waist_Circumference": waist_circ,
  "Resting_Heart_Rate": rest_hr,
  "Maximum_Heart_Rate": max_hr,
  "Exercise_Induced_Angina": ex_angina,
  "Resting_ECG": ecg,
  "ST_Depression": st_dep,
  "ST_Slope": st_slope,
  "Major_Vessels": major_vessels,
  "Thalassemia": thalassemia,
  "Smoking_Status": smoking,
  "Alcohol_Consumption": alcohol,
  "Physical_Activity": activity,
  "Daily_Steps": steps,
  "Family_History": family_hist,
  "Diabetes": diabetes,
  "Hypertension": hypertension,
  "Chronic_Kidney_Disease": ckd,
  "Stress_Level": stress,
  "Sleep_Hours": sleep,
  "Diet_Quality": diet,
}

current_signature = repr(tuple(input_data.items()))

if assess_clicked:
  raw_df = pd.DataFrame([input_data])
  fe_df = engineer_medical_features(raw_df)
  proba = pipeline.predict_proba(fe_df)[0, 1]

  st.session_state["cardio_last_signature"] = current_signature
  st.session_state["cardio_last_proba"] = float(proba)

show_result = (
  st.session_state.get("cardio_last_signature") == current_signature
  and "cardio_last_proba" in st.session_state
)

if show_result:
  proba = float(st.session_state["cardio_last_proba"])
  st.markdown(
    "<div style='height: 1rem;'></div>",
    unsafe_allow_html=True,
  )
  st.markdown('<div class="risk-card">', unsafe_allow_html=True)
  st.markdown(
    """
    <div class="risk-title">Heart Disease Risk Assessment</div>
    <div class="risk-sub">Exact probability returned by the model pipeline for the current input set.</div>
    """,
    unsafe_allow_html=True,
  )
  st.markdown(
    f"""
    <div class="risk-value">
      <div class="score">{proba * 100:.2f}%</div>
      <div class="status">Threshold: 0.40</div>
    </div>
    <div class="risk-track">
      <div class="risk-fill {'high' if proba >= 0.40 else 'low'}" style="width: {max(0.0, min(float(proba), 1.0)) * 100:.2f}%;"></div>
    </div>
    """,
    unsafe_allow_html=True,
  )

  if proba >= 0.40:
    st.markdown(
      """
      <div class="result-pill high">⚠️ Elevated Risk Detected</div>
      """,
      unsafe_allow_html=True,
    )
  else:
    st.markdown(
      """
      <div class="result-pill low">✅ Lower Predicted Risk</div>
      """,
      unsafe_allow_html=True,
    )
  st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-shell">', unsafe_allow_html=True)
section_header(
  "📘",
  "About the Model",
  "Evaluation metrics shown below are model-level values and are not patient-specific.",
)
st.markdown(
  """
  <div class="model-grid">
    <div class="model-card">
      <div class="kicker">Model</div>
      <div class="metric">XGBoost</div>
      <div class="hint">The existing trained pipeline is preserved exactly.</div>
    </div>
    <div class="model-card">
      <div class="kicker">Cross-Validation Accuracy</div>
      <div class="metric">84.26%</div>
      <div class="hint">Reported as a model evaluation metric.</div>
    </div>
    <div class="model-card">
      <div class="kicker">Cross-Validation Recall</div>
      <div class="metric">87.67%</div>
      <div class="hint">Reported as a model evaluation metric.</div>
    </div>
    <div class="model-card">
      <div class="kicker">Cross-Validation F1 / ROC-AUC</div>
      <div class="metric">86.45% / 93.24%</div>
      <div class="hint">Reported as model evaluation metrics only.</div>
    </div>
  </div>
  """,
  unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
  """
  <div class="section-shell">
    <div class="disclaimer">
      This tool is an educational machine-learning project and is not a medical diagnosis. Please consult a qualified healthcare professional for medical advice.
    </div>
  </div>
  """,
  unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)