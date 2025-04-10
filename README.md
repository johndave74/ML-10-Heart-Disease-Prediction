# ML-10-Heart-Disease-Prediction
This project uses supervised machine learning techniques to predict the presence of heart disease in patients based on various health parameters. The dataset contains features like age, gender, impulse, etc., and the target variable indicates the presence or absence of heart disease.

# ❤️ Heart Disease Prediction using Machine Learning

![image](https://github.com/user-attachments/assets/186bc4f6-755c-489e-8839-efe3e0598273)

Cardiovascular illnesses (CVDs) are the major cause of death worldwide. CVDs include coronary heart disease, cerebrovascular disease, rheumatic heart disease, and other heart and blood vessel problems. According to the World Health Organization, 17.9 million people die each year. Heart attacks and strokes account for more than four out of every five CVD deaths, with one-third of these deaths occurring before the age of 70. A comprehensive database for factors that contribute to a heart attack has been constructed.

This project uses supervised machine learning techniques to predict the presence of heart disease in patients based on various health parameters. The dataset contains features like age, cholesterol levels, chest pain types, etc., and the target variable indicates the presence or absence of heart disease.

## 📁 Dataset

The main purpose here is to collect characteristics of Heart Attack or factors that contribute to it.
The size of the dataset is 1319 samples, which have nine fields, where eight fields are for input fields and one field for an output field.

The dataset used is named `Heart Attack.csv`. It contains medical information related to patients and whether they have a heart condition (`class` column).

> ⚠️ **Download**: [Heart Disease](https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset)

### Sample Columns:
- `age`: Age of the patient
- `gender`: Gender (1 = male; 0 = female)
- `impulse`: heart rate
- `pressure height`: systolic BP
- `pressure low`: diastolic BP
- `glucose`: blood sugar
- `kcm`: CK-MB
- `troponin`: test-troponin
- `class`: Target (positive, negative)

---

## 🔍 Project Workflow

### 1. **Data Loading & Preprocessing**
- Loaded the dataset using `pandas`
- Checked for null values and general data structure
- Split the data into features (`X`) and labels (`y`)
- Used `LabelEncoder` to transform the `class` column into numeric format

### 2. **Train-Test Split**
- Used `train_test_split` from `sklearn.model_selection` to split the dataset
- 80% used for training and 20% for testing

---

## 🤖 Machine Learning Models

### 1. **Logistic Regression**
- Used L1 penalty and `liblinear` solver
- Evaluated performance using accuracy

### 2. **Decision Tree Classifier**
- Hyperparameters:
  - `max_depth=5`
  - `min_samples_leaf=2`
  - `min_samples_split=5`
  - `criterion='gini'`
- Visualized the tree using `plot_tree`
- Extracted and plotted feature importances

### 3. **Random Forest Classifier**
- Used 100 decision trees (`n_estimators=100`)
- Evaluated model accuracy
- Computed and visualized feature importances

---

## 📊 Evaluation Metric
All models were evaluated using **Accuracy Score**, but you can extend this to include:
- Confusion Matrix
- Precision, Recall, F1-Score
- ROC-AUC Curve

---

## 📈 Feature Importance
Feature importance was extracted from Decision Trees and Random Forests to understand which medical indicators contributed most to predictions. This is key in healthcare as it offers interpretability and supports clinical decisions.

---

## 📎 Visualizations
- Bar charts for feature importance
- Decision tree plot for interpretability

---

## 📦 Libraries Used

```bash
pandas
numpy
matplotlib
seaborn
sklearn
```

---

## 🔥 Future Improvements

- Add cross-validation for more robust evaluation
- Include ROC-AUC, Precision, Recall metrics
- Implement more models (e.g., XGBoost, SVM)
- Build a simple front-end app using Streamlit or Flask for user input
- Deploy model using platforms like Heroku or Hugging Face Spaces

---

## 🧠 Key Takeaways

- Machine Learning models can help in early diagnosis of heart diseases
- Feature importance gives insights into which parameters are medically significant
- Interpretability is crucial in healthcare ML applications

---

## 🧑‍💻 Author

**John David**  
*Data Scientist | ML Enthusiast | Healthcare AI Explorer*

---

## 📬 Feedback & Contributions

Feel free to fork, raise issues, or suggest improvements. Contributions are always welcome!
