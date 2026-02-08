import streamlit as st
import joblib
import os
import pandas as pd

st.set_page_config(page_title="Library Book Demand Prediction", layout="centered")

st.title("📚 University Library Book Demand Prediction")
st.write("Enter book details to predict demand level (High / Medium / Low).")

# Input fields
subject = st.selectbox(
    "Subject",
    ["Computer Science", "Mathematics", "Physics", "Management", "Biology", "Economics"]
)

semester = st.slider("Semester", 1, 8, 1)
past_borrow_count = st.number_input("Past Borrow Count", min_value=0, max_value=500, value=10)
course_relevance = st.selectbox("Course Relevance", [0, 1])

# Map subject to number (must match training encoding later)
subject_mapping = {
    "Computer Science": 0,
    "Mathematics": 1,
    "Physics": 2,
    "Management": 3,
    "Biology": 4,
    "Economics": 5
}

if st.button("Predict Demand"):
    model_path = "../models/best_model.pkl"

    if not os.path.exists(model_path):
        st.error("Model file not found. Please train the model first.")
    else:
        model = joblib.load(model_path)

        input_data = pd.DataFrame([[
            subject_mapping[subject],
            semester,
            past_borrow_count,
            course_relevance
        ]], columns=["subject", "semester", "past_borrow_count", "course_relevance"])

        prediction = model.predict(input_data)[0]

        # Map prediction back to label (we will align this later properly)
        label_mapping = {0: "Low", 1: "Medium", 2: "High"}
        predicted_label = label_mapping.get(prediction, "Unknown")

        st.success(f"📊 Predicted Demand Level: **{predicted_label}**")
