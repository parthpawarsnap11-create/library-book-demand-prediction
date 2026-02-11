1. Problem Definition (Clarity & Relevance)


The problem addressed in this project is predicting the demand level of university library books as High, Medium, or Low using academic and historical features such as school, subject, semester, past borrow count, and course relevance. This helps libraries plan inventory, allocate budgets better, and avoid overstocking or shortages.

---

 2. Project Architecture (End-to-End ML Pipeline)



The project is implemented as a complete end-to-end ML pipeline with four main components:

Data Generation using generate_data.py

Model Training and Evaluation using train_model.py

Model Saving as best_model.pkl

Deployment using a Streamlit dashboard (app.py)

---

 3. Folder Structure & Software Engineering Practices



The project follows a clean and modular structure:

data/ for dataset generation

scripts/ for model training

models/ for saved trained models

dashboard/ for the Streamlit UI

requirements.txt for dependencies

README.md for documentation
This structure follows good ML and software engineering practices.

---

 4. Version Control (GitHub)



The entire project is maintained in a GitHub repository with proper folder structure, commits, and documentation. This ensures version control, reproducibility, and easy collaboration.

---

 5. Models Used & Model Selection


I trained three machine learning models:

Logistic Regression

Decision Tree

Random Forest
I compared them using Accuracy and F1-score and selected the best performing model automatically. The best model is saved as best_model.pkl and used in the dashboard.


 6. Evaluation Metrics



I used Accuracy and F1-score. Accuracy tells the overall correctness of predictions, and F1-score balances precision and recall, which is important when class distributions may be uneven. This helps in choosing a more reliable model.

---

 7. Model Usage in Application



In the Streamlit app, the trained model is loaded using joblib. User inputs are converted into numerical format, passed to the model, and the output class (0, 1, or 2) is mapped to Low, Medium, or High demand.

---

 8. UI / Application Layer (Streamlit Dashboard)



I used Streamlit to build a simple and interactive dashboard where users can select school, subject, semester, past borrow count, and course relevance, and get a real-time prediction. This converts the ML model into a usable application.

---

 9. Maintenance & Timeline (VERY IMPORTANT FOR RUBRIC)


The model can be retrained periodically when new library data is available. The project is designed so that the data generation and training scripts can be rerun, and a new best model can replace the old one. This makes the system maintainable and updatable over time.

---

 10. Limitations & Future Scope



Currently, the project uses synthetic data and predicts only demand categories. In the future, this can be extended to use real library data, predict exact quantities, add analytics dashboards, and automate retraining.

---

This project demonstrates a complete end-to-end machine learning pipeline with data generation, model training, evaluation, model selection, deployment using Streamlit, and version control, aligned with real-world ML system design practices.
