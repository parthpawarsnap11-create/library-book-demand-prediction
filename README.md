# 📚 Library Book Demand Prediction System

![Python](https://img.shields.io/badge/Python-3.14-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> An end-to-end Machine Learning project that predicts the demand level of university library books as **High**, **Medium**, or **Low**.

## 📌 Problem Statement
University libraries struggle to manage book inventory efficiently. This system predicts book demand using:
- School & Subject
- Semester
- Past borrow count
- Course relevance

## 🏗️ Project Structure
```
library-book-demand-prediction/
├── dashboard/
│   └── app.py
├── data/
│   ├── generate_data.py
│   └── library_data.csv
├── models/
│   ├── best_model.pkl
│   └── model_metrics.json
├── scripts/
│   └── train_model.py
├── requirements.txt
└── README.md
```

## ⚙️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data processing |
| Scikit-Learn | ML model training |
| Streamlit | Interactive dashboard |

## 🚀 How to Run
1. Clone the repo
2. Run `pip install -r requirements.txt`
3. Run `python scripts/train_model.py`
4. Run `streamlit run dashboard/app.py`

## 💡 Features
- ✅ Predicts demand as High / Medium / Low
- ✅ Interactive Streamlit dashboard
- ✅ Clean modular code structure

## 👨‍💻 Author
**Parth Pawar** — [GitHub](https://github.com/parthpawarsnap11-create)
