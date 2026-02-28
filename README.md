🏦 Student Loan Default Predictor

Predicts whether a student will repay or default on an education loan.
Directly applicable to fintech companies like Bajaj Finance, HDFC, CRED.

📊 Dataset
5,000 synthetic student loan records (self-created with real-world bank logic)
Class distribution: 78% Repay vs 22% Default (realistic imbalance)

⚖️ Key Challenge
Imbalanced data handled using SMOTE

🤖 Models Compared
| Model | ROC-AUC | Recall |
|-------|---------|--------|
| XGBoost | Best | High |
| Random Forest | Good | Good |
| Logistic Regression | Baseline | Medium |

📏 Why Not Accuracy?
Accuracy is misleading for imbalanced data — used Recall, F1, and ROC-AUC instead

🛠️ Libraries Used
Python, XGBoost, Scikit-learn, imbalanced-learn, Pandas, Seaborn

🚀 How to Run
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn
python student_loan_default_predictor.py
