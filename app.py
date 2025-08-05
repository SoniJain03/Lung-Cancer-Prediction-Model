import streamlit as st
import pandas as pd
from joblib import load

# Load the trained model
model = load("modellung.joblib")

# App title
st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")
st.title("🩺 Lung Cancer Prediction App")
st.markdown("Predict the likelihood of lung cancer based on symptoms and personal details.")

# Sidebar About Section
with st.sidebar:
    st.title("About")
    st.markdown("""
    This app uses a K-Nearest Neighbors (KNN) machine learning model  
    to predict the likelihood of **lung cancer** based on user inputs such as:
    
    - Age and Gender  
    - Lifestyle habits (smoking, alcohol)  
    - Symptoms (wheezing, coughing, fatigue, etc.)
    
     Built with **Streamlit**  
     Model trained in **Python using Scikit-learn**
    
    **Disclaimer:** This is not a medical diagnosis.  
    Consult a healthcare professional for real medical advice.
    """)

# Input form
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 50)
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
    anxiety = st.selectbox("Anxiety", ["Yes", "No"])
    peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"])
    chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
    fatigue = st.selectbox("Fatigue", ["Yes", "No"])
    allergy = st.selectbox("Allergy", ["Yes", "No"])
    wheezing = st.selectbox("Wheezing", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol Consuming", ["Yes", "No"])
    coughing = st.selectbox("Coughing", ["Yes", "No"])
    shortness_of_breath = st.selectbox("Shortness of Breath", ["Yes", "No"])
    swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["Yes", "No"])
    chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])

    submit = st.form_submit_button("Predict")

# Helper function to map Yes/No and Gender to binary values
def map_input(val):
    return 1 if val == "Yes" or val == "Male" else 0

# When form is submitted
if submit:
    input_data = pd.DataFrame([[
        map_input(gender),
        age,
        map_input(smoking),
        map_input(yellow_fingers),
        map_input(anxiety),
        map_input(peer_pressure),
        map_input(chronic_disease),
        map_input(fatigue),
        map_input(allergy),
        map_input(wheezing),
        map_input(alcohol),
        map_input(coughing),
        map_input(shortness_of_breath),
        map_input(swallowing_difficulty),
        map_input(chest_pain)
    ]], columns=[
        "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
        "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE ", "ALLERGY ",
        "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH",
        "SWALLOWING DIFFICULTY", "CHEST PAIN"
    ])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("YES High risk of lung cancer.")
    else:
        st.success("NO risk of lung cancer.")
