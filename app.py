import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("datasets/student_data.csv")

# Features and target
X = data[['Attendance','StudyHours','InternalMarks','Assignments']]
y = data['FinalMarks']

# Train model
model = LinearRegression()
model.fit(X,y)

# App title
st.title("Student Performance Predictor")

st.write("Enter student details to predict final marks.")

# User inputs
attendance = st.slider("Attendance (%)",0,100,75)
study_hours = st.slider("Study Hours per Day",0,10,3)
internal_marks = st.slider("Internal Marks",0,20,15)
assignments = st.slider("Assignment Marks",0,20,15)

# Prediction button
if st.button("Predict Final Marks"):

    prediction = model.predict([[attendance,study_hours,internal_marks,assignments]])

    st.success(f"Predicted Final Marks: {round(prediction[0],2)}")