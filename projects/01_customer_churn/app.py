
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# 2. LOAD THE TRAINED MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "customer_churn_model.pkl"


@st.cache_resource
def load_model():
    artifact = joblib.load(MODEL_PATH)

    # Our saved artifact contains the pipeline and threshold
    if isinstance(artifact, dict):
        model = artifact["pipeline"]
        threshold = float(artifact["threshold"])
        feature_columns = artifact["feature_columns"]

    else:
        # Fallback if a plain sklearn pipeline was saved
        model = artifact
        threshold = 0.50
        feature_columns = None

    return model, threshold, feature_columns


try:
    model, threshold, feature_columns = load_model()

except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.title("📊 Customer Churn Prediction System")

st.write(
    "This application uses a machine learning model "
    "to estimate whether a telecom customer is likely "
    "to leave the service."
)

st.info(
    f"Model decision threshold: {threshold:.2f}"
)

st.divider()


# --------------------------------------------------
# 4. CUSTOMER INPUT FORM
# --------------------------------------------------

st.header("Enter Customer Information")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Personal Information")

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
            step=1
        )

        st.subheader("Phone Services")

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        if phone_service == "No":
            multiple_lines = "No phone service"
            st.selectbox(
                "Multiple Lines",
                ["No phone service"],
                disabled=True
            )
        else:
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes"]
            )

    with col2:

        st.subheader("Internet Services")

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        if internet_service == "No":

            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"

            st.caption(
                "Internet add-on services are unavailable "
                "when Internet Service is No."
            )

        else:

            internet_options = ["No", "Yes"]

            online_security = st.selectbox(
                "Online Security",
                internet_options
            )

            online_backup = st.selectbox(
                "Online Backup",
                internet_options
            )

            device_protection = st.selectbox(
                "Device Protection",
                internet_options
            )

            tech_support = st.selectbox(
                "Tech Support",
                internet_options
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                internet_options
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                internet_options
            )

    st.divider()

    st.subheader("Contract and Billing Information")

    col3, col4 = st.columns(2)

    with col3:

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col4:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=100000.0,
            value=840.0,
            step=10.0
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# 5. PREPARE CUSTOMER DATA
# --------------------------------------------------

if submitted:

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    customer = pd.DataFrame([customer_data])

    # Check that the input columns match model training
    if feature_columns is not None:

        missing_features = set(feature_columns) - set(customer.columns)
        extra_features = set(customer.columns) - set(feature_columns)

        if missing_features or extra_features:
            st.error(
                f"Input feature mismatch. "
                f"Missing: {missing_features}. "
                f"Extra: {extra_features}."
            )
            st.stop()

        customer = customer[feature_columns]

    # --------------------------------------------------
    # 6. MAKE PREDICTION
    # --------------------------------------------------

    try:

        churn_probability = model.predict_proba(customer)[0][1]

        prediction = int(churn_probability >= threshold)

        st.divider()
        st.header("Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )

        with result_col2:

            st.metric(
                "Decision Threshold",
                f"{threshold * 100:.0f}%"
            )

        st.progress(float(churn_probability))

        if prediction == 1:

            st.error(
                "High Churn Risk: The model predicts "
                "that this customer may leave the service."
            )

            st.write(
                "Consider reviewing this customer's "
                "service experience and possible retention options."
            )

        else:

            st.success(
                "Lower Churn Risk: The model predicts "
                "that this customer is less likely to leave."
            )

        st.caption(
            "This is a model-based estimate, not a certainty. "
            "A human should review important retention decisions."
        )

        with st.expander("View submitted customer information"):
            st.dataframe(customer, use_container_width=True)

    except Exception as error:

        st.error(f"Prediction failed: {error}")

