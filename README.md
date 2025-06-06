# Loan Approval Classification

## 📌 Project Overview
This project applies classification techniques to predict whether a loan applicant will be approved or rejected based on demographic, financial, and behavioral features. The goal is to support lending institutions in making data-driven decisions that improve risk assessment and reduce loan defaults.

## 🎯 Objectives
- Identify demographic and financial patterns linked to loan defaults.
- Determine if credit score is a reliable predictor of loan approval.
- Build, evaluate, and compare multiple classification models.
- Provide actionable recommendations to stakeholders based on model insights.

## 🧠 Business Context
In the financial sector, accurately predicting an applicant's creditworthiness is essential to reduce non-performing loans. This model aids in distinguishing low-risk applicants from high-risk ones using historical data.

## 🗃️ Dataset
- **Source**: [Kaggle - Loan Approval Classification Data](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data)
- **Size**: ~45,000 records
- **Features**: Age, Gender, Education, Income, Employment Experience, Credit Score, Loan Intent, Loan Amount, etc.
- **Target Variable**: `loan_status` (1 = Approved, 0 = Rejected)

## ⚙️ Technologies Used
- Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)
- Jupyter Notebook
- SMOTE for handling class imbalance
- RFECV for feature selection
- Logistic Regression, Decision Tree, XGBoost for modeling
- GridSearchCV for hyperparameter tuning

## 🧪 Modeling Process
1. **Data Cleaning**: Checked for missing values, duplicates, and categorical inconsistencies.
2. **Exploratory Data Analysis**: Visualized distributions and relationships between variables and loan status.
3. **Feature Engineering**: Encoded categorical variables and scaled numeric features.
4. **Feature Selection**: Applied Recursive Feature Elimination with Cross-Validation (RFECV).
5. **Modeling**:
    - Logistic Regression (baseline and tuned)
    - Decision Tree
    - XGBoost
6. **Evaluation Metrics**:
    - Accuracy
    - F1-Score
    - ROC-AUC
    - Confusion Matrix

## 📊 Results Summary
| Model              | Accuracy | F1 (Approved) | F1 (Rejected) | AUC   |
|-------------------|----------|---------------|----------------|-------|
| Logistic Regression (Tuned) | 89%      | 76%           | 93%            | 0.953 |
| Decision Tree      | 92%      | 79%           | 95%            | 0.965 |
| XGBoost            | 93%      | 80%           | 96%            | 0.967 |

## 📈 Key Insights
- Previous loan defaults and loan-to-income ratio are the strongest indicators.
- Credit score alone does not guarantee approval.
- Oversampling improved model fairness but introduced some overfitting.
- Decision Tree and XGBoost outperformed Logistic Regression in recall for the minority class (approved loans).

## 💡 Recommendations
- Implement XGBoost or Decision Tree models in production.
- Tune model thresholds based on organizational risk tolerance.
- Consider additional data sources for further improvements.
- Regularly retrain the model with new data.

## 🚀 Future Work
- Deploy the model as an API using Flask/FastAPI.
- Integrate with real-time loan application systems.
- Expand to multiclass predictions (e.g., high-risk, medium-risk, low-risk).

## 📂 Project Structure
