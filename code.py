# Install dependencies (optional if running locally)
# !pip install pandas numpy seaborn matplotlib scikit-learn

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

# Load dataset from URL provided in readme file.
# Example for local run: df = pd.read_csv("heart_disease.csv")
df = pd.read_csv("your_dataset_url.csv")  # Replace with actual link or local file

# Basic dataset info
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nData types:\n", df.info())
print("\nStatistical Summary:\n", df.describe())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Class distribution
sns.countplot(x='condition', data=df)
plt.title('Distribution of Heart Disease Cases')
plt.xlabel('Condition (0 = No Disease, 1 = Disease)')
plt.ylabel('Count')
plt.show()

# Feature-target split
X = df.drop('condition', axis=1)
y = df['condition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ROC Curve and AUC
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.show()

print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# Feature importance (coefficients)
coefficients = model.coef_[0]
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})
importance_df['AbsCoeff'] = importance_df['Coefficient'].abs()
importance_df.sort_values(by='AbsCoeff', ascending=True, inplace=True)

# Bar plot of coefficients
plt.figure(figsize=(8, 6))
plt.barh(importance_df['Feature'], importance_df['Coefficient'])
plt.title('Feature Coefficients (Logistic Regression)')
plt.xlabel('Coefficient Value')
plt.grid(True)
plt.tight_layout()
plt.show()
