
import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature names
model = joblib.load("burnout_model.pkl")
features = joblib.load("burnout_features.pkl")

# Web app title
st.title("Work From Home Burnout Prediction")
st.write("Enter your daily work details to predict burnout risk.")

# User inputs
day_type = st.selectbox(
    "Day Type",
    ["Weekday", "Weekend"]
)

work_hours = st.number_input(
    "Work Hours",
    min_value=0.0,
    max_value=24.0,
    value=8.0
)

screen_time_hours = st.number_input(
    "Screen Time Hours",
    min_value=0.0,
    max_value=24.0,
    value=8.0
)

meetings_count = st.number_input(
    "Meetings Count",
    min_value=0,
    max_value=20,
    value=3
)

breaks_taken = st.number_input(
    "Breaks Taken",
    min_value=0,
    max_value=20,
    value=3
)

after_hours_work = st.selectbox(
    "After-hours Work?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

task_completion_rate = st.number_input(
    "Task Completion Rate",
    min_value=0.0,
    max_value=110.0,
    value=80.0
)

# Predict button
if st.button("Predict Burnout Risk"):

    user_data = pd.DataFrame([{
        "day_type": day_type,
        "work_hours": work_hours,
        "screen_time_hours": screen_time_hours,
        "meetings_count": meetings_count,
        "breaks_taken": breaks_taken,
        "after_hours_work": after_hours_work,
        "sleep_hours": sleep_hours,
        "task_completion_rate": task_completion_rate
    }])

    # Convert categorical data into numeric columns
    user_data = pd.get_dummies(
        user_data,
        columns=["day_type"],
        dtype=int
    )

    # Match training feature names and order
    user_data = user_data.reindex(
        columns=features,
        fill_value=0
    )

    # Make prediction
    prediction = model.predict(user_data)[0]

    st.subheader("Prediction Result")
    st.write("Predicted Burnout Risk:", prediction)

    if prediction == "High":
        st.warning("The model predicts a high burnout risk.")
    elif prediction == "Medium":
        st.info("The model predicts a medium burnout risk.")
    else:
        st.success("The model predicts a low burnout risk.")
