import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model, scaler and feature names
with open('loan_default_model.pkl', 'rb') as f:
    model = pickle.load(f)
# rb = read binary mode
# loads our trained XGBoost model

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
# loads our fitted scaler

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
# loads column names

# App title
st.title("🏦 Loan Default Prediction")
st.write("Enter customer details to predict default risk")

# Input fields
st.header("Customer Details")

col1, col2 = st.columns(2)
# creates 2 side by side columns

with col1:
    loan_amnt = st.number_input("Loan Amount ($)", 
                    min_value=500, 
                    max_value=40000, 
                    value=10000)
    int_rate = st.slider("Interest Rate (%)", 
                    min_value=5.0, 
                    max_value=30.0, 
                    value=12.0)
    annual_inc = st.number_input("Annual Income ($)", 
                    min_value=10000, 
                    max_value=500000, 
                    value=60000)
    dti = st.slider("Debt to Income Ratio", 
                    min_value=0.0, 
                    max_value=50.0, 
                    value=15.0)
    term = st.selectbox("Loan Term", 
                    options=[36, 60])

with col2:
    grade = st.selectbox("Loan Grade", 
                    options=['A','B','C','D','E','F','G'])
    emp_length = st.slider("Employment Length (years)", 
                    min_value=0, 
                    max_value=10, 
                    value=5)
    fico_range_low = st.slider("FICO Score", 
                    min_value=580, 
                    max_value=850, 
                    value=700)
    revol_util = st.slider("Revolving Utilization (%)", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=40.0)
    home_ownership = st.selectbox("Home Ownership", 
                    options=['RENT', 'MORTGAGE', 'OWN'])

# Predict button
if st.button("🔍 Predict Default Risk"):
# runs when user clicks button

    # encode grade
    grade_map = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6}
    grade_encoded = grade_map[grade]

    # create input dataframe with zeros
    input_dict = {col: 0 for col in feature_names}
    # fills all features with 0 first

    # fill known values
    input_dict['loan_amnt'] = loan_amnt
    input_dict['int_rate'] = int_rate
    input_dict['annual_inc'] = annual_inc
    input_dict['dti'] = dti
    input_dict['term'] = term
    input_dict['grade'] = grade_encoded
    input_dict['emp_length'] = emp_length
    input_dict['fico_range_low'] = fico_range_low
    input_dict['revol_util'] = revol_util
    input_dict['loan_to_income'] = loan_amnt / annual_inc
    input_dict['installment_to_income'] = (loan_amnt/12) / (annual_inc/12)
    input_dict['fico_avg'] = fico_range_low

    # handle home ownership encoding
    if home_ownership == 'MORTGAGE':
        input_dict['home_ownership_MORTGAGE'] = 1
    elif home_ownership == 'OWN':
        input_dict['home_ownership_OWN'] = 1
    elif home_ownership == 'RENT':
        input_dict['home_ownership_RENT'] = 1

    # convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # scale input
    input_scaled = scaler.transform(input_df)

    # predict probability
    prob = model.predict_proba(input_scaled)[0][1]
    # [0] = first row
    # [1] = probability of default

    # show result
    st.header("🎯 Prediction Result")

    if prob >= 0.5:
        st.error(f"⚠️ HIGH RISK — Default Probability: {prob:.1%}")
    else:
        st.success(f"✅ LOW RISK — Default Probability: {prob:.1%}")

    # show probability bar
    st.progress(float(prob))
    st.write(f"Default Probability: **{prob:.1%}**")