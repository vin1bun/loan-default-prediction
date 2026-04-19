# 🏦 Loan Default Prediction

## 📌 Project Overview
An end-to-end Machine Learning project that predicts 
loan default risk using LendingClub data. 
Built to help financial institutions identify 
high-risk borrowers before loan disbursement.

## 🎯 Business Problem
LendingClub loses millions annually due to loan defaults. 
This model predicts which customers are likely to default,
enabling proactive risk management.

## 📊 Dataset
- **Source:** LendingClub Loan Data (Kaggle)
- **Size:** 200,000 rows × 38 features
- **Target:** Loan Default (1) vs No Default (0)
- **Class Imbalance:** 80:20 ratio

## 🔧 Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, SHAP
- **Deployment:** Streamlit
- **Environment:** Google Colab

## 📈 Project Pipeline
| Step | Task |
|------|------|
| 1 | Data Loading & Setup |
| 2 | Business Understanding |
| 3 | Exploratory Data Analysis |
| 4 | Data Cleaning |
| 5 | Feature Engineering |
| 6 | Preprocessing (Scaling + SMOTE) |
| 7 | Model Building |
| 8 | Evaluation |
| 9 | SHAP Explainability |
| 10 | Streamlit Deployment |

## 🤖 Models Trained
| Model | AUC-ROC |
|-------|---------|
| Logistic Regression | 0.671 |
| Random Forest | 0.591 |
| XGBoost (Final) | 0.743 |

## 🏆 Final Model Performance
- **Model:** XGBoost with Optuna Tuning
- **AUC-ROC:** 0.743
- **Recall (Defaulters):** 68%
- **PR AUC:** 0.425

## 💡 Key Findings
- Interest rate is the strongest predictor of default
- 60-month loans default significantly more than 36-month
- DTI and loan grade are strong risk indicators
- SHAP confirmed EDA findings on feature importance

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py


```## 📁 Project Structure
loan-default-prediction/
├── app.py                    # Streamlit web app
├── lendingclub.ipynb         # Full project notebook
├── loan_default_model.pkl    # Trained XGBoost model
├── scaler.pkl                # Fitted StandardScaler
└── feature_names.pkl         # Feature column names
## 👨‍💻 Author
**Vineet**
Aspiring Data Scientist | Delhi NCR



## 📁 Project Structure## 📁 Project Structure
