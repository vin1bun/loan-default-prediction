# 🏦 Loan Default Prediction


![Streamlit][(https://img.shields.io/badge/Streamlit-Live-red)](https://loan-default-prediction-l4joskzu8jcyxia6ftuucm.streamlit.app/) 


## 📌 Project Overview

Lending companies lose crores every year due to loan 
defaults. This project builds an end-to-end machine 
learning system that predicts which borrowers are 
likely to default — BEFORE the loan is approved.

The model doesn't just predict — it explains WHY 
a customer is flagged as high risk using SHAP 
explainability, making it ready for real-world 
banking environments where regulatory compliance 
requires explainable decisions.

🔗 **[Live Demo — Try it here](https://loan-default-prediction-l4joskzu8jcyxia6ftuucm.streamlit.app/)**

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Source | LendingClub (Kaggle) |
| Size | 200,000 rows |
| Features | 38 (selected from 151) |
| Target | Loan Default (1) vs No Default (0) |
| Class Imbalance | 80:20 ratio |

---

## 🗺️ Project Pipeline

| Step | Task | Key Decision |
|------|------|-------------|
| 1 | Data Loading | 200k rows to avoid RAM crash |
| 2 | Business Understanding | Identified target variable and leaky columns |
| 3 | EDA | 7 visualizations — found int_rate as strongest predictor |
| 4 | Data Cleaning | Removed outliers (DTI > 100), fixed encodings |
| 5 | Feature Engineering | Created loan_to_income, installment_to_income, fico_avg |
| 6 | Preprocessing | StandardScaler + SMOTE on train only |
| 7 | Model Building | LR → RF → XGBoost with Optuna tuning |
| 8 | Evaluation | AUC-ROC, Precision, Recall, F1, PR Curve |
| 9 | Explainability | SHAP bar + dot plots |
| 10 | Deployment | Live Streamlit web app on Streamlit Cloud |

---

## 🤖 Model Comparison

| Model | AUC-ROC | Recall (Defaulters) |
|-------|---------|-------------------|
| Logistic Regression | 0.671 | 0.64 |
| Random Forest | 0.591 | 0.25 |
| XGBoost (Final) | **0.743** | **0.68** |

> XGBoost won because of boosting — each tree learns 
> from previous mistakes. Optuna ran 50 trials to find 
> optimal hyperparameters.

---

## 🏆 Final Model Performance

| Metric | Score |
|--------|-------|
| AUC-ROC | **0.743** |
| Recall (Defaulters) | **0.68** |
| Precision (Defaulters) | 0.34 |
| F1 Score (Defaulters) | 0.45 |
| PR AUC | 0.425 |

> **Why AUC-ROC and not Accuracy?**
> Accuracy is misleading for imbalanced data — 
> a model predicting everyone as no default gets 
> 80% accuracy but zero business value. AUC-ROC 
> measures the model's ability to separate 
> defaulters from non-defaulters across all 
> thresholds.
> 
## 💡 Key Findings from EDA + SHAP

📊 EDA Findings:
→ Grade A borrowers: 6% default rate
→ Grade G borrowers: 59% default rate
→ Defaulters median int_rate: 14.5% vs 11% (non-defaulters)
→ DTI gap: 21 (defaulters) vs 18 (non-defaulters)
🔍 SHAP Findings:
→ int_rate    → strongest predictor 🔥
→ term        → 60 month loans 2x riskier than 36 month
→ sub_grade   → strong risk signal
→ dti         → high debt burden = more defaults
→ fico_score  → higher score = safer borrower

---

## ⚙️ Key Technical Decisions

**Why SMOTE?**
80:20 class imbalance meant the model would predict 
everyone as safe and still get 80% accuracy. SMOTE 
creates synthetic minority class samples — applied 
only on training data to prevent leakage.

**Why scale_pos_weight?**
XGBoost parameter set to 3.99 (ratio of majority to 
minority class) — tells the model to pay 4x more 
attention to defaulters.

**Why Optuna?**
Automated 50-trial Bayesian optimization found 
optimal parameters (n_estimators=74, 
learning_rate=0.034, max_depth=6) — improving 
AUC-ROC from 0.674 to 0.743.

**Why SHAP?**
In regulated banking environments, loan rejections 
must be explainable. SHAP provides feature-level 
explanations for every individual prediction.

---

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/vin1bun/loan-default-prediction.git
cd loan-default-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run loan_default_app/app.py
```

---

## 📁 Project Structure
loan-default-prediction/
├── loan_default_app/
│   ├── app.py                    # Streamlit web app
│   ├── loan_default_model.pkl    # Trained XGBoost model
│   ├── scaler.pkl                # Fitted StandardScaler
│   └── feature_names.pkl         # Feature column names
├── loandefault.ipynb             # Full project notebook
├── requirements.txt              # Dependencies
└── README.md                     # This file

---
---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP |
| Deployment | Streamlit, Streamlit Cloud |
| Version Control | Git, GitHub |

---

## 🔮 Future Improvements
→ Join all 7 LendingClub tables for richer features
(expected AUC-ROC improvement: 0.743 → 0.80+)
→ Add 5-fold cross validation for robust evaluation
→ Build ensemble model (XGBoost + LightGBM + CatBoost)
→ Add data drift monitoring for production deployment
→ Migrate from Streamlit to FastAPI for production API
---

## 👨‍💻 Author

**Vineet**
Aspiring Data Scientist | Delhi NCR

🔗 GitHub: [github.com/vin1bun](https://github.com/vin1bun)
🔗 LinkedIn: [linkedin.com/in/vineetprakash03](https://linkedin.com/in/vineetprakash03)
 🔗 Live Demo: [Click here](https://loan-default-prediction-l4joskzu8jcyxia6ftuucm.streamlit.app/)

---

⭐ If you found this project helpful, please give it a star!
