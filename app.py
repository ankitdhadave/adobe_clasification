
import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load model
# ==============================
@st.cache_resource
def load_model():
    return joblib.load("conversion_model.pkl")

model = load_model()

st.set_page_config(page_title="Conversion Prediction", layout="centered")

st.title("📊 Adobe Conversion Prediction")
st.write("Predict whether a customer will convert")

# ==============================
# User Inputs
# ==============================
st.sidebar.header("Customer Details")

def user_input():
    return pd.DataFrame([{
        'Company_Size': st.sidebar.selectbox(
            "Company Size",
            ['Small', 'Medium', 'Large'],
            key='company_size'
        ),

        'Subscription_Plan': st.sidebar.selectbox(
            "Subscription Plan",
            ['Basic', 'Standard', 'Premium'],
            key='subscription_plan'
        ),

        'Region': st.sidebar.selectbox(
            "Region",
            ['APAC', 'EMEA', 'NA'],
            key='region'
        ),

        'Industry': st.sidebar.selectbox(
            "Industry",
            ['IT', 'Finance', 'Healthcare', 'Other'],
            key='industry'
        ),

        'Device_Type': st.sidebar.selectbox(
            "Device Type",
            ['Desktop', 'Mobile'],
            key='device_type'
        ),

        'Marketing_Channel': st.sidebar.selectbox(
            "Marketing Channel",
            ['Email', 'Ads', 'Organic', 'Referral'],
            key='marketing_channel'
        ),

        'Avg_Session_Time_mins': st.sidebar.number_input(
            "Avg Session Time (mins)",
            0.0, 300.0, 15.0,
            key='avg_session_time'
        ),

        'Cloud_Storage_Usage_GB': st.sidebar.number_input(
            "Cloud Storage Usage (GB)",
            0.0, 10000.0, 100.0,
            key='cloud_storage'
        ),

        'Files_Created': st.sidebar.number_input(
            "Files Created",
            0, 100000, 50,
            key='files_created'
        ),

        'Website_Visits': st.sidebar.number_input(
            "Website Visits",
            0, 1000, 10,
            key='website_visits'
        ),

        'Satisfaction_Score': st.sidebar.slider(
            "Satisfaction Score",
            1, 10, 7,
            key='satisfaction_score'
        ),

        'Last_Login_Days_Ago': st.sidebar.number_input(
            "Last Login Days Ago",
            0, 365, 5,
            key='last_login_days'
        ),

        'Login_Frequency_per_Week': st.sidebar.number_input(
            "Login Frequency / Week",
            0, 50, 5,
            key='login_frequency'
        ),

        'Email_Click_Through_Rate': st.sidebar.number_input(
            "Email CTR",
            0.0, 1.0, 0.15,
            key='email_ctr'
        ),

        'Days_Active_in_Trial': st.sidebar.number_input(
            "Days Active in Trial",
            0, 30, 10,
            key='days_active_trial'
        ),

        'Discount_Offered_USD': st.sidebar.number_input(
            "Discount Offered (USD)",
            0.0, 500.0, 0.0,
            key='discount_offered'
        ),

        'Subscription_Fee_USD': st.sidebar.number_input(
            "Subscription Fee (USD)",
            0.0, 500.0, 29.0,
            key='subscription_fee'
        ),

        'Support_Tickets_Raised': st.sidebar.number_input(
            "Support Tickets Raised",
            0, 50, 0,
            key='support_tickets'
        ),

        'Competitor_Subscription_Price': st.sidebar.number_input(
            "Competitor Price (USD)",
            0.0, 500.0, 25.0,
            key='competitor_price'
        ),

        'Engagement_Score': st.sidebar.number_input(
            "Engagement Score",
            0.0, 100.0, 60.0,
            key='engagement_score'
        )
    }])

input_df = user_input()

st.subheader("Input Data")
st.write(input_df)

# ==============================
# Prediction
# ==============================
if st.button("Predict Conversion"):
    probability = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.metric("Conversion Probability", f"{probability:.2%}")

    if prediction == 1:
        st.success("✅ Likely to Convert")
    else:
        st.error("❌ Unlikely to Convert")
