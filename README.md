Library Book Demand Prediction System

This project implements a minimal end-to-end Machine Learning system to predict the demand level of university library books as **High**, **Medium**, or **Low** based on historical and academic features.

The goal of this project is not only to build a predictive model, but also to demonstrate a **production-like ML pipeline** including data generation, model training, evaluation, deployment via a dashboard, and version control.

---

Problem Statement

University libraries often face challenges in managing book inventory efficiently. Some books are in very high demand during certain semesters, while others remain underutilized. This system helps predict the **demand level of books** using features such as:

- Subject
- Semester
- Past borrow count
- Course relevance

This prediction can help libraries in:
- Better inventory planning
- Budget allocation
- Deciding number of copies to keep for each book



 Project Structure


---

 Data

- The dataset is generated using a **Python script** (`data/generate_data.py`)
- This ensures **reproducibility** and avoids using static datasets
- Features include:
  - Subject
  - Semester
  - Past borrow count
  - Course relevance
- Target:
  - Demand level (High / Medium / Low)

---

Models Used

The training pipeline trains and evaluates **three traditional ML models**:

1. Logistic Regression  
2. Random Forest Classifier  
3. Decision Tree Classifier  

Models are compared using:
- Accuracy
- F1-score  

The **best performing model** is automatically selected and saved for use in the dashboard.

---

Dashboard

A Streamlit-based dashboard is provided to:

- Display a simple user interface
- Take book details as input
- Predict the demand level using the trained model

This makes the system **prediction-ready** and usable by non-technical users.

---
How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
