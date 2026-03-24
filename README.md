 Problem Statement
University libraries struggle to manage book inventory efficiently. This system predicts book demand using academic and historical features such as:

School & Subject
Semester
Past borrow count
Course relevance


Project Structure
library-book-demand-prediction/
│
├── dashboard/
│   └── app.py                  # Streamlit web app
│
├── data/
│   ├── generate_data.py        # Data generation script
│   └── library_data.csv        # Dataset
│
├── models/
│   ├── best_model.pkl          # Trained ML model
│   └── model_metrics.json      # Model evaluation metrics
│
├── scripts/
│   └── train_model.py          # Model training script
│
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md

Tech Stack
ToolPurposePythonCore languagePandas & NumPyData processingScikit-LearnML model trainingStreamlitInteractive dashboardMatplotlib/SeabornData visualization

How to Run
1. Clone the repository
bashgit clone https://github.com/parthpawarsnap11-create/library-book-demand-prediction.git
cd library-book-demand-prediction
2. Install dependencies
bashpip install -r requirements.txt
3. Train the model
bashpython scripts/train_model.py
4. Launch the dashboard
bashstreamlit run dashboard/app.py

 Model Performance
MetricScoreAccuracySee model_metrics.jsonModel TypeBest selected via cross-validationClassesHigh / Medium / Low

 Features

✅ Predicts book demand as High / Medium / Low
✅ Interactive Streamlit dashboard
✅ Automated data generation
✅ Model training with evaluation metrics
✅ Clean, modular code structure


Author
Parth Pawar
GitHub

📄 License
This project is for academic purposes.
