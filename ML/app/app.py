import streamlit as st
import pandas as pd
import joblib
import os

# Load trained model
model_path = os.path.join(os.path.dirname('C:/Users/lakshya.vashisth/Documents/Assignments/ML/app/credit_risk_model.pkl'), "credit_risk_model.pkl")
model = joblib.load(model_path)

# UI Header
st.set_page_config(page_title="Credit Risk Prediction App", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #353434; padding: 20px;}
        .stButton>button {background-color: #007BFF; color: white; font-size: 16px; padding: 10px 20px; border-radius: 5px;}
        .stTextInput, .stNumberInput, .stSelectbox, .stRadio, .stSlider {margin-bottom: 15px;}
        h1, h2, h3, h4, h5, h6 {font-weight: bold;}
        h1 {font-size: 3rem;}
        h2 {font-size: 2.5rem;}
        h3 {font-size: 2rem;}
        .section-heading {font-size: 1.75rem; font-weight: bold; margin-top: 10px; margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# UI Title
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="font-size: 2.5rem; color: #007BFF;">Credit Risk Prediction App</h1>
        <p style="font-size: 1.2rem; color: #555;">Predict credit risk using machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

def percent_income_ratio(income, amount):
    if income != 0:
        result = income / amount
        return result
    else:
        return None

status_mapping = {"Yes": 1, "No": 0}

# Input form
def user_input():
    st.header("Enter Applicant Information")  
    
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income", min_value=0, value=50000, max_value=1000000)
    emp_length = st.number_input("Employment Length (years)", min_value=0, value=5)
    amount = st.number_input("Loan Amount", min_value=0, value=10000)
    rate = st.number_input("Interest Rate", min_value=0.0, value=10.0)
    cred_length = st.number_input("Credit Length (in years)", min_value=0.0, value=1.0, max_value=99.0)
    purpose = st.selectbox("Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    home_ownership = st.selectbox("Home Ownership status", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    percent_income = st.number_input("Ratio (Output)", value=percent_income_ratio(income, amount), disabled=True)
    status = st.selectbox("Loan Approval Status", options=list(status_mapping.keys()))

    data = {
        'Age': age,
        'Income': income,
        'Emp_length': emp_length,
        'Amount': amount,
        'Rate': rate,
        'Cred_length': cred_length,
        'Intent': purpose,
        'Home': home_ownership,
        'Percent_income': percent_income,
        'Status': status_mapping[status]   
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input()

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader("Prediction")
    st.write("**Credit Risk:**", "High" if prediction[0] == 1 else "Low")
    st.write("**Probability of High Risk:**", prediction_proba[0][1])