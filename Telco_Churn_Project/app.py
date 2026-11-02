
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a telecom customer is likely to leave the company."
)

MODEL_PATH = Path("best_model.pkl")
SCALER_PATH = Path("scaler.pkl")
FEATURES_PATH = Path("feature_columns.pkl")

if not all(p.exists() for p in [
    MODEL_PATH, SCALER_PATH, FEATURES_PATH
]):
    st.error(
        "Model files are missing. Run the notebook "
        "and save the model, scaler, and feature columns first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)

with st.form("customer_form"):
    st.subheader("Customer Information")

    c1, c2 = st.columns(2)

    with c1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )
        senior = st.selectbox(
            "Senior Citizen", [0, 1]
        )
        partner = st.selectbox(
            "Has Partner?", ["Yes", "No"]
        )
        dependents = st.selectbox(
            "Has Dependents?", ["Yes", "No"]
        )
        tenure = st.slider(
            "Tenure (months)", 0, 72, 12
        )
        phone = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with c2:
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0, value=70.0
        )
        total = st.number_input(
            "Total Charges",
            min_value=0.0, value=840.0
        )

    # Additional categorical features
    online_security = st.selectbox(
        "Online Security", ["No", "Yes", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup", ["No", "Yes", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection", ["No", "Yes", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support", ["No", "Yes", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", ["No", "Yes", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", ["No", "Yes", "No internet service"]
    )

    submitted = st.form_submit_button(
        "Predict Churn", type="primary"
    )

if submitted:
    raw = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    categorical_columns = [
        "gender", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]

    categories = {
        "gender": ["Female", "Male"],
        "Partner": ["No", "Yes"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": [
            "No", "No phone service", "Yes"
        ],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": [
            "No", "No internet service", "Yes"
        ],
        "OnlineBackup": [
            "No", "No internet service", "Yes"
        ],
        "DeviceProtection": [
            "No", "No internet service", "Yes"
        ],
        "TechSupport": [
            "No", "No internet service", "Yes"
        ],
        "StreamingTV": [
            "No", "No internet service", "Yes"
        ],
        "StreamingMovies": [
            "No", "No internet service", "Yes"
        ],
        "Contract": [
            "Month-to-month", "One year", "Two year"
        ],
        "PaperlessBilling": ["No", "Yes"],
        "PaymentMethod": [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    }

    customer = pd.DataFrame([raw])

    for col in categorical_columns:
        customer[col] = pd.Categorical(
            customer[col],
            categories=categories[col]
        )

    customer = pd.get_dummies(
        customer,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    customer = customer.reindex(
        columns=feature_columns,
        fill_value=0
    )

    customer_scaled = scaler.transform(customer)
    prediction = model.predict(customer_scaled)[0]

    st.divider()
    st.subheader("Prediction Result")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(
            customer_scaled
        )[0][1]

        st.metric(
            "Estimated Churn Probability",
            f"{probability:.1%}"
        )

    if prediction == 1:
        st.warning(
            "This customer is predicted to churn."
        )
    else:
        st.success(
            "This customer is predicted to stay."
        )

    st.caption(
        "This is a model prediction, not a guarantee "
        "of future customer behavior."
    )
