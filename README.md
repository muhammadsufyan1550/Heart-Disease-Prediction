#  Heart Disease Prediction - Machine Learning Project

### 📌 Objective:
Predict the presence of heart disease in patients using machine learning based on 13 clinical features.

---

### 📁 Dataset:
- **Source**: UCI Heart Disease Dataset
- **Rows**: 297
- **Features**: 13 + 1 target (`target`)
- **Target**: 1 = heart disease, 0 = no disease

---

### 🔧 ML Pipeline:
- Data visualization: heatmap, class countplot
- Model: Logistic Regression
- Train/Test Split: 80/20
- Metrics: Accuracy, ROC-AUC, F1-Score

---

### 📊 Evaluation Results:
- **Accuracy**: 73.33%
- **Precision**: 70%
- **Recall**: 75%
- **ROC AUC Score**: 0.837
- **Confusion Matrix**:
- [[23 9]
[ 7 21]]

---

### 🔍 Feature Importance (Top Coefficients):
| Feature | Impact |
|---------|--------|
| `sex`   | +1.5   |
| `ca`    | +1.2   |
| `thal`  | +0.8   |
| `fbs`   | −0.8   |
| `exang` | +0.6   |

> Features like age and cholesterol had minimal influence

---

### 🚀 Tools Used:
- Python, Pandas, Matplotlib, Seaborn, scikit-learn, Google Colab
