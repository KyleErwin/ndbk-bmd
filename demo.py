import requests
import streamlit as st

st.set_page_config(page_title="Bank Marketing")
st.title("Bank Marketing")

JOB_OPTIONS = {
    "Admin": "admin",
    "Blue-collar": "blue-collar",
    "Entrepreneur": "entrepreneur",
    "Housemaid": "housemaid",
    "Management": "management",
    "Retired": "retired",
    "Self-employed": "self-employed",
    "Services": "services",
    "Student": "student",
    "Technician": "technician",
    "Unemployed": "unemployed",
    "Unknown": "unknown",
}

MARITAL_OPTIONS = {
    "Married": "married",
    "Single": "single",
    "Divorced": "divorced",
}

EDUCATION_OPTIONS = {
    "Primary": "primary",
    "Secondary": "secondary",
    "Tertiary": "tertiary",
    "Unknown": "unknown",
}

CONTACT_OPTIONS = {
    "Cell": "cell",
    "Email": "email",
}

POUTCOME_OPTIONS = {
    "Failure": "failure",
    "Other": "other",
    "Success": "success",
    "Unknown": "unknown",
}

MONTH_OPTIONS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

with st.form("prediction_form"):
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Customer Profile")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        job = st.selectbox("Job", options=list(JOB_OPTIONS.keys()))
        marital = st.selectbox("Marital status", options=list(MARITAL_OPTIONS.keys()))
        balance = st.number_input("Balance", value=0.0, step=100.0)
        education = st.selectbox("Education", options=list(EDUCATION_OPTIONS.keys()))
        default = st.checkbox("Has credit in default?")
        housing = st.checkbox("Has housing loan?")
        loan = st.checkbox("Has personal loan?")

    with col_right:
        st.markdown("#### Campaign Information")
        previous = st.number_input(
            "Previous number of campaign contacts", min_value=0, value=0
        )
        pdays = st.number_input(
            "Days since last contact (-1 = never)", min_value=-1, value=-1
        )
        poutcome = st.selectbox(
            "Previous campaign outcome", options=list(POUTCOME_OPTIONS.keys())
        )
        month = st.selectbox(
            "Current campaign contact month", options=MONTH_OPTIONS, index=4
        )
        contact = st.selectbox(
            "Current campaign contact type", options=list(CONTACT_OPTIONS.keys())
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "age": age,
        "balance": balance,
        "default": default,
        "housing": housing,
        "loan": loan,
        "campaign": 1,
        "job": JOB_OPTIONS[job],
        "marital": MARITAL_OPTIONS[marital],
        "education": EDUCATION_OPTIONS[education],
        "previous": previous,
        "month": month,
        "pdays": pdays,
        "contact": CONTACT_OPTIONS[contact],
        "poutcome": POUTCOME_OPTIONS[poutcome],
    }

    API_URL = "https://app-production-c465.up.railway.app"

    with st.spinner("Getting prediction..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            response.raise_for_status()
            result = response.json()
            st.success(f"Prediction: **{result['prediction']}**")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
