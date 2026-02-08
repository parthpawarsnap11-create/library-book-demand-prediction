import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Load data
data_path = "../data/library_data.csv"
df = pd.read_csv(data_path)

# Encode categorical features
le_subject = LabelEncoder()
df["subject"] = le_subject.fit_transform(df["subject"])

le_target = LabelEncoder()
df["demand_level"] = le_target.fit_transform(df["demand_level"])

# Features and target
X = df[["subject", "semester", "past_borrow_count", "course_relevance"]]
y = df["demand_level"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

best_model = None
best_f1 = 0

print("Training models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"{name} -> Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model

# Save best model
os.makedirs("../models", exist_ok=True)
joblib.dump(best_model, "../models/best_model.pkl")

print("\nBest model saved to models/best_model.pkl")

