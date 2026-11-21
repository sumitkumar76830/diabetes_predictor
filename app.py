
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="VitalIQ | Diabetes AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model_files():
    model = joblib.load("diabetes_model.pkl")
    imputer = joblib.load("diabetes_imputer.pkl")
    features = joblib.load("feature_columns.pkl")

    features = [
        col for col in features
        if col != "Diabetes_Yes"
    ]

    return model, imputer, features


try:
    model, imputer, features = load_model_files()
except Exception as error:
    st.error(f"Model files could not be loaded: {error}")
    st.stop()


# ==========================================
# CUSTOM CSS - COMPLETELY NEW DESIGN
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #10191d;
    --panel: #1a282d;
    --panel2: #22343a;
    --cyan: #00d6b2;
    --orange: #ffab64;
    --text: #f0f7f5;
    --muted: #91a7ab;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1500px;
    padding: 1.5rem 3rem 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: var(--bg);
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
}

p, label {
    color: #c3d1d2;
}

/* TOP BRAND */

.brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 27px;
    font-weight: 800;
    color: white;
    letter-spacing: -1px;
}

.brand span {
    color: var(--cyan);
}

.topline {
    border-bottom: 1px solid #34474c;
    padding-bottom: 15px;
    margin-bottom: 25px;
}

/* HERO */

.hero {
    background:
        radial-gradient(circle at 85% 20%, #28534c 0%, transparent 38%),
        linear-gradient(115deg, #1a3034, #1a282d 65%, #253b3b);
    border: 1px solid #35534f;
    border-radius: 22px;
    padding: 34px;
    margin: 18px 0 26px;
}

.hero-tag {
    display: inline-block;
    background: #254b45;
    color: #00e1bd;
    border: 1px solid #34766b;
    border-radius: 30px;
    padding: 6px 13px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.3px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    color: white;
    font-size: clamp(30px, 4vw, 47px);
    font-weight: 800;
    line-height: 1.15;
    margin-top: 20px;
}

.hero-title span {
    color: var(--cyan);
}

.hero-desc {
    color: #a8babc;
    font-size: 15px;
    margin-top: 13px;
    max-width: 650px;
    line-height: 1.8;
}

/* STATUS */

.status {
    background: #1b2c30;
    border: 1px solid #34494c;
    border-radius: 14px;
    padding: 18px;
    min-height: 105px;
}

.status-label {
    color: #91a7ab;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.3px;
}

.status-value {
    color: white;
    font-size: 23px;
    font-weight: 800;
    margin-top: 8px;
}

.status-note {
    color: #91a7ab;
    font-size: 12px;
}

/* PANELS */

.panel {
    background: var(--panel);
    border: 1px solid #33464b;
    border-radius: 18px;
    padding: 23px;
    margin-bottom: 18px;
}

.panel-heading {
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 21px;
    font-weight: 700;
}

.panel-sub {
    color: var(--muted);
    font-size: 12px;
    margin-top: 5px;
}

/* FORM INPUTS */

.stNumberInput input,
.stTextInput input {
    background: #263a40 !important;
    color: white !important;
    border: 1px solid #486067 !important;
    border-radius: 10px !important;
    min-height: 43px;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: #263a40 !important;
    color: white !important;
    border-color: #486067 !important;
    border-radius: 10px !important;
}

.stSelectbox div[data-baseweb="select"] span {
    color: white !important;
}

[data-testid="stWidgetLabel"] p {
    color: #c9d8d9 !important;
    font-weight: 600;
    font-size: 13px;
}

/* BUTTONS */

.stButton button,
.stFormSubmitButton button {
    background: #00d6b2 !important;
    color: #09201d !important;
    border: 0 !important;
    border-radius: 10px !important;
    min-height: 47px;
    font-weight: 800;
    transition: .2s;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #61f0d6 !important;
    box-shadow: 0 5px 20px #00d6b233;
}

/* TOP NAVIGATION */

div[role="radiogroup"] {
    gap: 9px;
}

div[role="radiogroup"] label {
    background: #1c2b30;
    border: 1px solid #3a4e53;
    border-radius: 10px;
    padding: 10px 17px;
    color: white !important;
}

div[role="radiogroup"] label:has(input:checked) {
    background: #00d6b2;
    border-color: #00d6b2;
    color: #08201c !important;
}

div[role="radiogroup"] label p {
    color: inherit !important;
}

/* TABS */

.stTabs [data-baseweb="tab-list"] {
    background: #1b2b30;
    border-radius: 12px;
    padding: 7px;
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 9px;
    color: #b4c5c7;
    padding: 12px 20px;
}

.stTabs [aria-selected="true"] {
    background: #00d6b2 !important;
    color: #08201c !important;
}

.stTabs [aria-selected="true"] p {
    color: #08201c !important;
}

/* RESULT */

.result-panel {
    background:
        radial-gradient(circle at 90% 10%, #27645a, transparent 48%),
        linear-gradient(140deg, #203c3d, #1b292e);
    border: 1px solid #47776e;
    border-radius: 19px;
    padding: 26px;
}

.result-kicker {
    color: #9debdc;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.result-main {
    color: white;
    font-size: 27px;
    font-weight: 800;
    margin: 14px 0;
}

.result-caption {
    color: #b4c9c7;
    font-size: 13px;
    line-height: 1.7;
}

/* PROGRESS */

.stProgress > div > div > div {
    background: #00d6b2 !important;
}

/* METRICS */

[data-testid="stMetric"] {
    background: #1c2b30;
    border: 1px solid #3b5055;
    border-radius: 13px;
    padding: 15px;
}

[data-testid="stMetricValue"] {
    color: #00d6b2 !important;
}

/* DATAFRAME */

[data-testid="stDataFrame"] {
    border: 1px solid #34494c;
    border-radius: 12px;
}

/* FOOTER */

.footer {
    border-top: 1px solid #33464b;
    padding: 20px;
    text-align: center;
    color: #71898e;
    font-size: 12px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# TOP NAVIGATION
# ==========================================

top1, top2 = st.columns([2, 3])

with top1:
    st.markdown(
        '<div class="brand">VITAL<span>IQ.</span></div>',
        unsafe_allow_html=True
    )

with top2:
    st.markdown(
        '<div style="text-align:right;padding-top:8px;color:#00d6b2;font-size:12px;font-weight:700;">'
        '● AI ENGINE ONLINE &nbsp;&nbsp; | &nbsp;&nbsp; HEALTH ANALYTICS'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="topline"></div>', unsafe_allow_html=True)

page = st.radio(
    "MAIN MENU",
    ["Overview", "Patient Assessment", "Project Insights"],
    horizontal=True,
    label_visibility="collapsed"
)


# ==========================================
# REUSABLE HERO
# ==========================================

def show_hero(title, highlight, description, tag):

    st.markdown(f"""
    <div class="hero">
        <div class="hero-tag">{tag}</div>
        <div class="hero-title">{title}<br><span>{highlight}</span></div>
        <div class="hero-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def show_footer():
    st.markdown("""
    <div class="footer">
        VITALIQ HEALTH ANALYTICS &nbsp; • &nbsp;
        MACHINE LEARNING PROJECT
        <br><br>
        Academic demonstration only. Not a medical diagnosis.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 1 - OVERVIEW
# ==========================================

if page == "Overview":

    show_hero(
        "Understand your health.",
        "Predict with intelligence.",
        "An interactive diabetes classification platform that "
        "uses patient health parameters and a trained ML model.",
        "✦ INTELLIGENT HEALTHCARE PLATFORM"
    )

    st.markdown("### Platform Overview")

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        ("MODEL STATUS", "ONLINE", "Prediction engine loaded"),
        ("ALGORITHM", type(model).__name__, "Trained ML classifier"),
        ("INPUT FEATURES", str(len(features)), "Patient parameters"),
        ("PLATFORM", "VITALIQ AI", "Healthcare analytics")
    ]

    for col, item in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="status">
                <div class="status-label">{item[0]}</div>
                <div class="status-value">{item[1]}</div>
                <div class="status-note">{item[2]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([1.35, 1])

    with left:

        st.markdown("""
        <div class="panel">
            <div class="panel-heading">The Intelligence Behind Your Health</div>
            <div class="panel-sub">
                How the platform works
            </div>
            <br>
            <p>01 &nbsp; Patient Information</p>
            <p style="color:#91a7ab;font-size:13px;">
            Collect clinical and lifestyle information.
            </p>
            <hr>
            <p>02 &nbsp; Data Processing</p>
            <p style="color:#91a7ab;font-size:13px;">
            Prepare the patient data for machine learning.
            </p>
            <hr>
            <p>03 &nbsp; AI Classification</p>
            <p style="color:#91a7ab;font-size:13px;">
            The trained model produces a classification.
            </p>
            <hr>
            <p>04 &nbsp; Prediction Insights</p>
            <p style="color:#91a7ab;font-size:13px;">
            Display the estimated model output.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("""
        <div class="panel">
            <div class="panel-heading">Designed for clarity.</div>
            <div class="panel-sub">Your assessment workspace</div>
            <br>
            <h2 style="color:#00d6b2;">01</h2>
            <p>Simple patient data entry</p>
            <br>
            <h2 style="color:#ffab64;">02</h2>
            <p>Machine learning classification</p>
            <br>
            <h2 style="color:#00d6b2;">03</h2>
            <p>Clear and readable results</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("START NEW ASSESSMENT →", use_container_width=True):
        st.session_state["open_assessment"] = True
        st.rerun()

    show_footer()


# ==========================================
# PAGE 2 - PATIENT ASSESSMENT
# ==========================================

elif page == "Patient Assessment":

    show_hero(
        "Patient assessment.",
        "Your health profile.",
        "Enter the patient details below. The trained model "
        "will process the information and generate a prediction.",
        "✦ NEW PATIENT ANALYSIS"
    )

    left, right = st.columns([1.5, 0.85], gap="large")

    with left:

        st.markdown("""
        <div class="panel">
            <div class="panel-heading">Patient Data Entry</div>
            <div class="panel-sub">
                Complete all three sections before submitting.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("vital_patient_form"):

            tab1, tab2, tab3 = st.tabs([
                "01 · Personal",
                "02 · Clinical",
                "03 · Lifestyle"
            ])

            with tab1:

                st.markdown("#### Personal Information")

                p1, p2 = st.columns(2)

                with p1:
                    age = st.number_input(
                        "Age (years)",
                        min_value=1,
                        max_value=120,
                        value=30
                    )

                with p2:
                    gender = st.selectbox(
                        "Gender",
                        ["Female", "Male", "Unknown"]
                    )

                family = st.selectbox(
                    "Family History of Diabetes",
                    ["No", "Yes"]
                )

            with tab2:

                st.markdown("#### Clinical Measurements")

                p1, p2 = st.columns(2)

                with p1:
                    bmi = st.number_input(
                        "BMI",
                        10.0, 70.0, 25.0, step=0.1
                    )

                    glucose = st.number_input(
                        "Glucose Level",
                        40.0, 400.0, 100.0
                    )

                    bp = st.number_input(
                        "Blood Pressure",
                        40, 250, 120
                    )

                with p2:
                    cholesterol = st.number_input(
                        "Cholesterol",
                        50, 500, 180
                    )

                    insulin = st.number_input(
                        "Insulin",
                        0, 1000, 100
                    )

                    hba1c = st.number_input(
                        "HbA1c",
                        3.0, 20.0, 5.5, step=0.1
                    )

            with tab3:

                st.markdown("#### Lifestyle Information")

                smoking = st.selectbox(
                    "Smoking Status",
                    ["Current", "Former", "Never"]
                )

                activity = st.selectbox(
                    "Physical Activity",
                    ["High", "Low", "Moderate"]
                )

                st.caption(
                    "Please enter accurate information "
                    "for a meaningful model assessment."
                )

            st.write("")

            submitted = st.form_submit_button(
                "✦ ANALYZE PATIENT DATA",
                use_container_width=True
            )

    with right:

        st.markdown("""
        <div class="panel">
            <div class="panel-heading">Assessment Guide</div>
            <div class="panel-sub">Your analysis at a glance</div>
            <br>
            <div style="color:#00d6b2;font-size:12px;font-weight:800;">
                STEP 01
            </div>
            <p>Personal details</p>
            <hr>
            <div style="color:#ffab64;font-size:12px;font-weight:800;">
                STEP 02
            </div>
            <p>Clinical measurements</p>
            <hr>
            <div style="color:#00d6b2;font-size:12px;font-weight:800;">
                STEP 03
            </div>
            <p>Lifestyle information</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
            <div class="panel-heading">AI Model</div>
            <br>
            <div style="font-size:13px;color:#91a7ab;">ACTIVE ALGORITHM</div>
            <h3 style="color:#00d6b2 !important;">
            Random Forest / ML
            </h3>
            <div style="font-size:13px;color:#91a7ab;">
            Input parameters are processed through the
            saved preprocessing pipeline.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # RESULT SECTION

    if submitted:

        patient_values = {
            "Age": age,
            "BMI": bmi,
            "Glucose": glucose,
            "Blood_Pressure": bp,
            "Cholesterol": cholesterol,
            "Insulin": insulin,
            "HbA1c": hba1c,

            "Gender_Male": int(gender == "Male"),
            "Gender_Unknown": int(gender == "Unknown"),

            "Smoking_Status_Former": int(smoking == "Former"),
            "Smoking_Status_Never": int(smoking == "Never"),

            "Physical_Activity_Low": int(activity == "Low"),
            "Physical_Activity_Moderate": int(activity == "Moderate"),

            "Family_History_Yes": int(family == "Yes")
        }

        input_data = pd.DataFrame([patient_values])

        input_data = input_data.reindex(
            columns=features,
            fill_value=0
        )

        processed_data = pd.DataFrame(
            imputer.transform(input_data),
            columns=features
        )

        prediction = model.predict(processed_data)[0]

        st.markdown("---")
        st.markdown("## Prediction Report")

        r1, r2 = st.columns([1.15, 1])

        with r1:

            if prediction == 1:
                result = "Positive Classification"
                symbol = "!"
                description = (
                    "The model classified this patient "
                    "in the positive diabetes class."
                )
                result_color = "#ffab64"
            else:
                result = "Negative Classification"
                symbol = "✓"
                description = (
                    "The model classified this patient "
                    "in the negative diabetes class."
                )
                result_color = "#00d6b2"

            st.markdown(f"""
            <div class="result-panel">
                <div class="result-kicker">
                    VITALIQ · AI REPORT
                </div>
                <div style="font-size:55px;color:{result_color};">
                    {symbol}
                </div>
                <div class="result-main">{result}</div>
                <div class="result-caption">
                    {description}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:

            st.markdown("""
            <div class="panel">
                <div class="panel-heading">Patient Summary</div>
                <div class="panel-sub">
                    Submitted health information
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.metric("Patient Age", f"{age} years")
            st.metric("BMI", f"{bmi:.1f}")
            st.metric("Glucose", f"{glucose:.1f}")

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(processed_data)[0]

            class_list = list(model.classes_)
            positive_index = class_list.index(1)

            probability = probabilities[positive_index] * 100

            st.markdown("### Positive-Class Probability")

            st.metric(
                "Estimated probability",
                f"{probability:.2f}%"
            )

            st.progress(
                int(round(probability))
            )

        with st.expander("VIEW COMPLETE PATIENT REPORT"):

            report = pd.DataFrame({
                "Health Parameter": [
                    "Age", "Gender", "Family History",
                    "BMI", "Glucose", "Blood Pressure",
                    "Cholesterol", "Insulin", "HbA1c",
                    "Smoking Status", "Physical Activity"
                ],
                "Patient Value": [
                    age, gender, family, bmi, glucose,
                    bp, cholesterol, insulin, hba1c,
                    smoking, activity
                ]
            })

            st.dataframe(
                report,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇ Download Patient Report",
                report.to_csv(index=False),
                file_name="patient_report.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.warning(
            "Academic machine learning estimate only. "
            "This result is not a medical diagnosis. "
            "Consult a qualified healthcare professional."
        )

    show_footer()


# ==========================================
# PAGE 3 - PROJECT INSIGHTS
# ==========================================

elif page == "Project Insights":

    show_hero(
        "Behind the prediction.",
        "The technology.",
        "Explore the machine learning model, preprocessing "
        "workflow and tools used to build this project.",
        "✦ PROJECT INFORMATION"
    )

    st.markdown("### Technology Stack")

    t1, t2, t3, t4 = st.columns(4)

    tech = [
        ("PYTHON", "Core programming"),
        ("PANDAS", "Data processing"),
        ("SCIKIT-LEARN", "Machine learning"),
        ("STREAMLIT", "Interactive interface")
    ]

    for col, item in zip([t1, t2, t3, t4], tech):
        with col:
            st.markdown(f"""
            <div class="status">
                <div class="status-label">{item[0]}</div>
                <div class="status-value">✦</div>
                <div class="status-note">{item[1]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    st.markdown("### Model Details")

    m1, m2 = st.columns(2)

    with m1:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-heading">Classification Engine</div>
            <br>
            <div class="status-label">MODEL CLASS</div>
            <h2 style="color:#00d6b2 !important;">
                {type(model).__name__}
            </h2>
            <p style="color:#91a7ab;">
            Loaded from the trained Jupyter Notebook model.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-heading">Input Schema</div>
            <br>
            <h2 style="color:#ffab64 !important;">
                {len(features)} Features
            </h2>
            <p style="color:#91a7ab;">
            Patient information used by the classifier.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("VIEW MODEL INPUT FEATURES"):

        st.write(features)

    st.markdown("### Project Workflow")

    st.markdown("""
    <div class="panel">
        <h3>01 · Dataset Collection</h3>
        <p>Patient health data is loaded into the notebook.</p>
        <hr>
        <h3>02 · Data Preprocessing</h3>
        <p>Missing values and categorical data are handled.</p>
        <hr>
        <h3>03 · Model Training</h3>
        <p>A machine learning classifier is trained.</p>
        <hr>
        <h3>04 · Model Deployment</h3>
        <p>The trained model is connected to a Streamlit application.</p>
    </div>
    """, unsafe_allow_html=True)

    show_footer()
