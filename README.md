# Predict-the-Introverts-from-the-Extroverts-using-Machine-Learning

A machine learning web app that predicts whether a person is an Introvert or Extrovert based on their social behaviour patterns — built with XGBoost and Streamlit.

📸 Demo

Fill in 7 simple inputs → get a prediction with confidence score and feature breakdown.


🗂️ Project Structure
```
introvert-extrovert-classifier/
├── app.py                          # Streamlit app (UI + prediction logic)
├── requirements.txt                # Python dependencies
├── .gitignore
├── README.md
└── model_files/
    ├── xgbc_model.pkl              # Trained XGBoost model (download separately — see below)
    ├── feature_columns.json        # Ordered feature names used during training
    ├── label_map.json              # {0: "Introvert", 1: "Extrovert"}
    └── medians.json                # Training-set medians used as default UI values
```

⚙️ How It Works
The app accepts 7 user inputs and internally applies the same feature engineering pipeline used during model training:
```
Input                              Type
Time Spent Alone (hrs/day)         Numeric (0–10)
Stage Fear                         Dropdown (Yes / No)
Social Event Attendance            Numeric (0–10)
Going Outside (times/week)         Numeric (0–10)
Drained After Socializing          Dropdown (Yes / No)
Friends Circle Size                Numeric (0–20)
Post Frequency (posts/week)        Numeric (0–10)
```
Behind the scenes, app.py automatically computes 6 additional engineered features before passing data to the model:

- social_engagement = social events + going outside + post frequency
- introvert_score = time alone + stage fear + drained after socializing
- social_vs_alone = social engagement / (time alone + 1)
- friends_per_event = friends / (social events + 1)
- online_vs_offline = post frequency / (going outside + 1)
- alone_time_level = binned time alone into 4 levels [0–3]


🚀 Setup & Run Locally
1. Clone the repo
```bash
git clone https://github.com/GauravRawat9/Predict-the-Introverts-from-the-Extroverts-using-Machine-Learning.git
cd introvert-extrovert-classifier
```
2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Add the model file
Download xgbc_model.pkl from the Releases page (or from your Kaggle output) and place it inside the model_files/ folder.
The other three JSON files (feature_columns.json, label_map.json, medians.json) are already in the repo.

5. Run the app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

🤖 Model Details
```
Property                 Value
Algorithm                XGBoost Classifier
Tuning                   GridSearchCV (5-fold CV)
Hyperparameters tuned    n_estimators, max_depth, learning_rate
Target                   Personality → Introvert (0) / Extrovert (1)
Training notebook        Kaggle Notebook
```
📦 Why is xgbc_model.pkl not in the repo?
.pkl model files can exceed GitHub's 100MB file limit. The model is provided via the Releases tab instead. If your model file is small enough, you can remove the relevant line from .gitignore and commit it directly.

🛠️ Tech Stack
- Python 3.10+
- Streamlit — UI
- XGBoost — classification model
- scikit-learn — preprocessing & evaluation
- pandas / numpy — data handling
