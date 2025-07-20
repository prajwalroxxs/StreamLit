
import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("Titanic Survival Predictor")
st.write("Enter passenger details below:")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.slider("Siblings/Spouses Aboard", 0, 5, 0)
parch = st.slider("Parents/Children Aboard", 0, 5, 0)
fare = st.slider("Fare Paid", 0.0, 300.0, 32.0)

sex_encoded = 1 if sex == "Male" else 0
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])

if st.button("Predict"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.subheader("Prediction")
    st.write(f"The passenger would have **{'survived' if pred == 1 else 'not survived'}**.")

    st.subheader("Prediction Probabilities")
    st.bar_chart(proba)
