
import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('model.pkl')

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

# Input widgets
pclass = st.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
sex = st.radio('Sex', ['Male', 'Female'])
age = st.slider('Age', 1, 80, 30)
sibsp = st.slider('Siblings/Spouses aboard (SibSp)', 0, 5, 0)
parch = st.slider('Parents/Children aboard (Parch)', 0, 5, 0)
fare = st.slider('Fare ($)', 0.0, 300.0, 32.0)

# Preprocess input
sex_encoded = 1 if sex == 'Male' else 0
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])

# Make prediction
pred = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0]

# Display results
st.subheader("Prediction")
st.write(f"The passenger would have **{'survived' if pred == 1 else 'not survived'}**.")

st.subheader("Prediction Probabilities")
st.bar_chart(proba)
