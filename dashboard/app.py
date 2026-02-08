import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Library Book Demand Prediction",
    page_icon="📚",
    layout="wide"
)

# -------------------------
# Custom CSS for white + colorful UI
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #ffffff;
}
h1, h2, h3 {
    color: #2c3e50;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 200px;
    font-size: 16px;
}
.result-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f1f8ff;
    border-left: 6px solid #3498db;
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Title
# -------------------------
st.title("📚 University Library Book Demand Prediction")
st.write("Enter book details to predict demand level (High / Medium / Low)")

# -------------------------
# Load model
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please run: python scripts/train_model.py")
    st.stop()

model = joblib.load(model_path)

# -------------------------
# Inputs
# -------------------------
col1, col2 = st.columns(2)

with col1:
    subject = st.selectbox(
        "Subject",
        ["Computer Science", "Mathematics", "Physics", "Management", "Biology", "Economics"]
    )

    semester = st.slider("Semester", 1, 8, 3)

with col2:
    past_borrow_count = st.number_input("Past Borrow Count", min_value=0, max_value=100, value=10)
    course_relevance = st.selectbox("Course Relevance", [0, 1, 2, 3])

# Subject mapping (must match training)
subject_mapping = {
    "Computer Science": 0,
    "Mathematics": 1,
    "Physics": 2,
    "Management": 3,
    "Biology": 4,
    "Economics": 5
}

# -------------------------
# Predict
# -------------------------
if st.button("🔮 Predict Demand"):
    input_data = pd.DataFrame([[
        subject_mapping[subject],
        semester,
        past_borrow_count,
        course_relevance
    ]], columns=["subject", "semester", "past_borrow_count", "course_relevance"])

    prediction = model.predict(input_data)[0]

    demand_map = {0: "Low", 1: "Medium", 2: "High"}
    demand_text = demand_map.get(prediction, str(prediction))

    st.markdown(f"""
    <div class="result-card">
        📊 Predicted Demand Level: <span style="color:#e74c3c">{demand_text}</span>
    </div>
    """, unsafe_allow_html=True)

    st.success("✅ Prediction completed successfully!")

# -------------------------
# Analytics Section
# -------------------------
st.markdown("---")
st.header("📈 Analytics & Insights")

# Try loading dataset if exists
data_path = os.path.join(BASE_DIR, "data", "library_data.csv")

if os.path.exists(data_path):
    df = pd.read_csv(data_path)

    colA, colB = st.columns(2)

    with colA:
        st.subheader("📊 Demand Distribution")
        demand_counts = df["demand"].value_counts()

        fig1, ax1 = plt.subplots()
        ax1.bar(demand_counts.index.astype(str), demand_counts.values)
        ax1.set_xlabel("Demand Level")
        ax1.set_ylabel("Count")
        ax1.set_title("Demand Distribution")
        st.pyplot(fig1)

    with colB:
        st.subheader("🥧 Subject-wise Demand Share")
        subject_counts = df["subject"].value_counts()

        fig2, ax2 = plt.subplots()
        ax2.pie(subject_counts.values, labels=subject_counts.index, autopct="%1.1f%%")
        ax2.set_title("Subject Distribution")
        st.pyplot(fig2)

    st.subheader("📋 Sample Data")
    st.dataframe(df.head(20))

else:
    st.info("ℹ️ No dataset found for analytics. Run data generation script to enable charts.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("👤 **Author:** Parth Niraj Pawar | Vijaybhoomi University Project Dashboard")
