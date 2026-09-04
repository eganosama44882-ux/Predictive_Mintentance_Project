import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("random_forest_model.joblib")

st.title("Machine Failure Prediction")
st.write("Enter the machine parameters to predict whether a failure is likely.")

# Inputs
machine_type = st.selectbox(
    "Machine Type",
    [0, 1, 2]
)

air_temperature = st.number_input(
    "Air Temperature [K]",
    value=300.0
)

process_temperature = st.number_input(
    "Process Temperature [K]",
    value=310.0
)

rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    value=1500
)

torque = st.number_input(
    "Torque [Nm]",
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    value=100
)

if st.button("Predict Machine Failure"):

    new_machine = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    prediction = model.predict(new_machine)[0]

    if prediction == 1:
        st.error("⚠️ Machine Failure: YES")
    else:
        st.success("✅ Machine Failure: NO")

    probability = model.predict_proba(new_machine)[0][1]

    st.write(f"Failure Probability: {probability:.2%}")