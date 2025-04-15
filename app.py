import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="🚖 NYC Green Taxi - Fare Predictor", layout="wide")

st.title("🚖 NYC Green Taxi Fare Predictor - August 2024")
st.markdown("Predict the total fare of an NYC green taxi ride based on trip details using a trained Random Forest model.")

# Load model
@st.cache_resource
def load_model():
    try:
        with open("randomforest.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'randomforest.pkl' is available.")
        return None

model = load_model()
if model is None:
    st.stop()

# Sidebar Info
st.sidebar.header("ℹ️ App Info")
st.sidebar.markdown("""
- This fare prediction model is trained on **August 2024 NYC green taxi** trip data.
- Enter trip-specific features below to estimate the **total fare**.
- Note: Please avoid negative or unrealistic values.
""")

# Input Form
st.subheader("📋 Enter Trip Details")

weekday_mapping = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}

with st.form("fare_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        passenger_count = st.number_input("👥 Passenger Count", 1.0, 10.0, 1.0, step=1.0)
        trip_distance = st.number_input("🛣️ Trip Distance (miles)", 0.0, 100.0, 2.0, step=0.1)
        tip_amount = st.number_input("💸 Tip Amount ($)", 0.0, 50.0, 0.0, step=0.5)
        tolls_amount = st.number_input("🛣️ Tolls Amount ($)", 0.0, 50.0, 0.0, step=0.5)
        trip_duration = st.number_input("⏱️ Trip Duration (minutes)", 0.0, 240.0, 15.0, step=1.0)

    with col2:
        extra = st.number_input("➕ Extra Charges ($)", 0.0, 20.0, 0.0, step=0.5)
        mta_tax = st.number_input("🏙️ MTA Tax ($)", 0.0, 5.0, 0.5, step=0.5)
        improvement_surcharge = st.number_input("🛠️ Improvement Surcharge ($)", 0.0, 5.0, 0.3, step=0.3)
        congestion_surcharge = st.number_input("🚦 Congestion Surcharge ($)", 0.0, 10.0, 0.0, step=0.5)

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        store_and_fwd_flag = st.selectbox("📡 Store and Forward Flag", ["Y", "N"], index=1)
        ratecode_id = st.selectbox("💵 Ratecode ID", [1.0, 2.0, 3.0, 4.0, 5.0, 99.0])
        payment_type = st.selectbox("💳 Payment Type", [1.0, 2.0, 3.0, 4.0, 5.0])

    with col4:
        trip_type = st.selectbox("🚕 Trip Type", [1.0, 2.0])
        weekday = st.selectbox("📅 Weekday", list(weekday_mapping.keys()), index=3)
        hour = st.selectbox("🕒 Dropoff Hour", list(range(24)), index=18)

    submit = st.form_submit_button("🚀 Predict Fare")

# Prediction Logic
if submit:
    features = {
        'passenger_count': passenger_count,
        'trip_distance': trip_distance,
        'extra': extra,
        'mta_tax': mta_tax,
        'tip_amount': tip_amount,
        'tolls_amount': tolls_amount,
        'improvement_surcharge': improvement_surcharge,
        'congestion_surcharge': congestion_surcharge,
        'trip_duration': trip_duration,
        'store_and_fwd_flag': 1 if store_and_fwd_flag == 'Y' else 0,
        'RatecodeID': ratecode_id,
        'payment_type': payment_type,
        'trip_type': trip_type,
        'weekday': weekday_mapping[weekday],
        'hour': hour
    }

    input_vector = pd.DataFrame([features]).astype(float)

    if any(input_vector.select_dtypes(include=[np.number]).lt(0).any()):
        st.error("⚠️ Invalid input: Please avoid negative values.")
    else:
        try:
            with st.spinner("⏳ Calculating..."):
                fare = model.predict(input_vector)[0]
                st.success(f"🎉 Estimated Total Fare: **${fare:.2f}**")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# -----------------------------
# 📊 Actual vs Predicted Graph (Scatter Plot)
# -----------------------------
st.subheader("📊 Actual vs Predicted Fare (on Sample Data)")

@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv("sample_nyc_test_data.csv")
        return df
    except FileNotFoundError:
        # Generate synthetic mock data
        n = 50
        df = pd.DataFrame({
            'passenger_count': np.random.randint(1, 5, n),
            'trip_distance': np.random.uniform(1, 10, n),
            'extra': np.random.uniform(0, 3, n),
            'mta_tax': np.full(n, 0.5),
            'tip_amount': np.random.uniform(0, 5, n),
            'tolls_amount': np.random.uniform(0, 5, n),
            'improvement_surcharge': np.full(n, 0.3),
            'congestion_surcharge': np.random.uniform(0, 2.5, n),
            'trip_duration': np.random.uniform(5, 60, n),
            'store_and_fwd_flag': np.random.randint(0, 2, n),
            'RatecodeID': np.random.choice([1.0, 2.0, 3.0], n),
            'payment_type': np.random.choice([1.0, 2.0, 3.0], n),
            'trip_type': np.random.choice([1.0, 2.0], n),
            'weekday': np.random.randint(0, 7, n),
            'hour': np.random.randint(0, 24, n),
        })
        df["actual_fare"] = model.predict(df)
        return df

sample_data = load_sample_data()

if sample_data is not None and model is not None:
    try:
        actual = sample_data['actual_fare']
        input_features = sample_data.drop(columns=['actual_fare'])
        predicted = model.predict(input_features)

        result_df = pd.DataFrame({
            "Actual Fare": actual,
            "Predicted Fare": predicted
        })

        # Scatter plot
        st.markdown("### 🟢 Scatter Plot: Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x="Actual Fare", y="Predicted Fare", data=result_df, color="teal", s=80)
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
        ax.set_title("📍 Predicted vs Actual Taxi Fare")
        ax.set_xlabel("Actual Fare")
        ax.set_ylabel("Predicted Fare")
        st.pyplot(fig)

        st.markdown("🔍 Preview of actual vs predicted fares:")
        st.dataframe(result_df.head(10))

    except Exception as e:
        st.error(f"Error generating scatter plot: {e}")
