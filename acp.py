import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Student Alcohol Predictor",
    page_icon="🌿",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f5f8f7;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(135deg, #123c32, #24735f);
    padding: 40px;
    border-radius: 25px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 15px 40px rgba(20, 70, 55, 0.18);
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    font-size: 17px;
    color: #dff3eb;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #e3ebe7;
    box-shadow: 0 8px 25px rgba(0,0,0,0.04);
}

.stButton > button,
.stFormSubmitButton > button {
    background: #176b57;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-weight: 700;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #105342;
    color: white;
}

div[data-testid="stMetric"] {
    background: white;
    border-radius: 18px;
    padding: 15px;
    border: 1px solid #e3ebe7;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

MODEL_PATH = Path(__file__).parent / "student-alcohol-consumption.pkl"

if not MODEL_PATH.exists():
    st.error("student-alcohol-consumption.pkl was not found.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("Model could not be loaded.")
    st.code(str(e))
    st.stop()


# ---------------- FEATURES ----------------

FEATURE_COLUMNS = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "health",
    "absences"
]


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">

<div style="font-size:13px; letter-spacing:3px; font-weight:700;">
MACHINE LEARNING • STUDENT LIFESTYLE
</div>

<h1>🌿 Student Alcohol<br>Consumption Predictor</h1>

<p>
Enter student information and lifestyle details to predict
the weekend alcohol consumption score.
</p>

</div>
""", unsafe_allow_html=True)


# ---------------- METRICS ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Prediction Target", "Walc")

with col2:
    st.metric("Score Range", "1 - 5")

with col3:
    st.metric("Model", "Random Forest")


st.markdown("## 👤 Student Information")


# ---------------- FORM ----------------

with st.form("student_form"):

    tab1, tab2, tab3 = st.tabs(
        ["👤 Profile", "📚 Education & Family", "🌱 Lifestyle"]
    )

    # ---------------- PROFILE ----------------

    with tab1:

        col1, col2, col3 = st.columns(3)

        with col1:

            school = st.selectbox(
                "School",
                ["GP", "MS"],
                help="GP = Gabriel Pereira, MS = Mousinho da Silveira"
            )

            sex = st.selectbox(
                "Sex",
                ["F", "M"]
            )

            age = st.slider(
                "Age",
                15,
                22,
                17
            )

        with col2:

            address = st.selectbox(
                "Address",
                ["U", "R"],
                help="U = Urban, R = Rural"
            )

            famsize = st.selectbox(
                "Family Size",
                ["GT3", "LE3"],
                help="GT3 = greater than 3, LE3 = 3 or less"
            )

            pstatus = st.selectbox(
                "Parents Status",
                ["T", "A"],
                help="T = Together, A = Apart"
            )

        with col3:

            medu = st.selectbox(
                "Mother Education",
                [0, 1, 2, 3, 4]
            )

            fedu = st.selectbox(
                "Father Education",
                [0, 1, 2, 3, 4]
            )

            guardian = st.selectbox(
                "Guardian",
                ["mother", "father", "other"]
            )


    # ---------------- EDUCATION ----------------

    with tab2:

        col1, col2, col3 = st.columns(3)

        with col1:

            mjob = st.selectbox(
                "Mother Job",
                ["teacher", "health", "services", "at_home", "other"]
            )

            fjob = st.selectbox(
                "Father Job",
                ["teacher", "health", "services", "at_home", "other"]
            )

            reason = st.selectbox(
                "Reason for School Choice",
                ["course", "home", "reputation", "other"]
            )

        with col2:

            traveltime = st.selectbox(
                "Travel Time",
                [1, 2, 3, 4]
            )

            studytime = st.selectbox(
                "Study Time",
                [1, 2, 3, 4]
            )

            failures = st.selectbox(
                "Past Failures",
                [0, 1, 2, 3]
            )

        with col3:

            famrel = st.selectbox(
                "Family Relationship",
                [1, 2, 3, 4, 5]
            )

            schoolsup = st.selectbox(
                "Extra School Support",
                ["yes", "no"]
            )

            famsup = st.selectbox(
                "Family Support",
                ["yes", "no"]
            )


    # ---------------- LIFESTYLE ----------------

    with tab3:

        col1, col2, col3 = st.columns(3)

        with col1:

            paid = st.selectbox(
                "Extra Paid Classes",
                ["yes", "no"]
            )

            activities = st.selectbox(
                "Extra Activities",
                ["yes", "no"]
            )

            nursery = st.selectbox(
                "Nursery School",
                ["yes", "no"]
            )

            higher = st.selectbox(
                "Higher Education",
                ["yes", "no"]
            )

        with col2:

            internet = st.selectbox(
                "Internet Access",
                ["yes", "no"]
            )

            romantic = st.selectbox(
                "Romantic Relationship",
                ["yes", "no"]
            )

            freetime = st.selectbox(
                "Free Time",
                [1, 2, 3, 4, 5]
            )

            goout = st.selectbox(
                "Going Out",
                [1, 2, 3, 4, 5]
            )

        with col3:

            dalc = st.selectbox(
                "Workday Alcohol Consumption (Dalc)",
                [1, 2, 3, 4, 5]
            )

            health = st.selectbox(
                "Health",
                [1, 2, 3, 4, 5]
            )

            absences = st.number_input(
                "Absences",
                min_value=0,
                max_value=100,
                value=0
            )


    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.form_submit_button(
        "✨ Predict Weekend Alcohol Consumption",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------

if predict_button:

    input_data = {

        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": pstatus,
        "Medu": medu,
        "Fedu": fedu,
        "Mjob": mjob,
        "Fjob": fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": dalc,
        "health": health,
        "absences": absences
    }

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURE_COLUMNS
    )

    try:

        prediction = float(model.predict(input_df)[0])

        prediction = max(1, min(5, prediction))

        rounded_prediction = round(prediction)

        labels = {
            1: "Very Low",
            2: "Low",
            3: "Moderate",
            4: "High",
            5: "Very High"
        }

        st.markdown("---")

        st.markdown("## 📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Walc Score",
                f"{prediction:.2f} / 5"
            )

        with col2:

            st.metric(
                "Predicted Category",
                labels[rounded_prediction]
            )

        st.progress(
            rounded_prediction / 5
        )

        st.success(
            f"Predicted weekend alcohol consumption score: "
            f"**{prediction:.2f} / 5**"
        )

        st.info(
            "This prediction is generated by a machine-learning model "
            "and should be treated as an educational project result."
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.write(
            "The input columns must exactly match the columns "
            "used during model training."
        )

        st.code(str(e))


# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "Student Alcohol Consumption Prediction • Built with Python, "
    "Scikit-learn & Streamlit"
)