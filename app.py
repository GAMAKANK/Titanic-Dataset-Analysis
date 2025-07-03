import streamlit as st
from datetime import datetime
import pickle
import numpy as np
import os

st.set_page_config(page_title="Titanic Data Analysis", layout="wide", page_icon="⛴️")

# Load model
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

#Sidebar Inputs
st.sidebar.title("Titanic Survival Analysis")
passengerId = st.sidebar.number_input("Enter Passenger ID",min_value=0,max_value=1000,value=0)
pClass = int(st.sidebar.selectbox("Passenger Class" , ["1","2","3"]))
sex = st.sidebar.selectbox("Enter passenger's gender",["male","female"])
age = st.sidebar.number_input("Age",min_value=0,max_value=180,value=30)
sibSp = st.sidebar.number_input("Number of Siblings or Spouses Aboard the Titanic with the Passenger",min_value=0,max_value=1000,value=0)
parch = st.sidebar.number_input("Number of Parents or Children Aboard the Titanic with the Passenger",min_value=0,max_value=1000,value=0)
fare = st.sidebar.number_input("Enter the Fare",min_value=0.00, max_value=1000.00, value=250.00, step=0.01)
embarked = st.sidebar.selectbox("Embarked from",["C","S","Q"])

#on clicking the submit buttom storing it in session state
if "submitted" not in st.session_state:
    st.session_state.submitted = False

if st.sidebar.button("Analyze Return ->"):
    st.session_state.submitted = True

# Main Section
st.title("Titanic Survival Analysis")

st.subheader("Survival Insights")
st.markdown(f"""
- *Passenger ID*: {passengerId}
- *Passenger Class*: {pClass}
- *Passenger Gender*: {sex}
- *Passenger Age*: {age}
- *Number of Siblings or Spouses Aboard the Titanic with the Passenger*: {sibSp}
- *Number of Parents or Children Aboard the Titanic with the Passenger*: {parch}
- *Fare*: {fare}
- *Embarked From*: {embarked}
""")

if st.session_state.submitted:

    st.subheader("Analysis Results")

    #Converting according to trained model 
    male = 1 if sex == "male" else 0

    Q = 1 if embarked == "Q" else 0
    S = 1 if embarked == "S" else 0

    input_features = np.array([[passengerId,int(pClass),age,sibSp,parch,fare,male,Q,S]])

    prediction = model.predict(input_features)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_features)[0][1] * 100
    else:
        proba = None

    # Display result
    if prediction == 1:
        st.success(f"Prediction: Passenger would **Survive**" +
                   (f" ({proba:.2f}% confidence)" if proba else ""))
    else:
        st.error(f"Prediction: Passenger would **Not Survive**" +
                 (f" ({100 - proba:.2f}% confidence)" if proba else ""))




# Footer
st.markdown("---")
st.markdown("© 2025 Titanic Survival Analysis")