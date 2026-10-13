import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from common import load_csv

print("Worksheet 6.6 - Exercise 2: Diabetes risk classification")
df = load_csv("diabetes.csv", "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
df = df.rename(columns={"Outcome": "disease"})
X, y = df.drop(columns="disease"), df["disease"]
zero_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X[zero_missing] = X[zero_missing].replace(0, np.nan)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
pred, prob = model.predict(X_test), model.predict_proba(X_test)[:, 1]
print("Confusion matrix:\n", metrics.confusion_matrix(y_test, pred))
print("Accuracy:", round(metrics.accuracy_score(y_test, pred), 3))
print("ROC-AUC:", round(metrics.roc_auc_score(y_test, prob), 3))
print(metrics.classification_report(y_test, pred, zero_division=0))
patient = pd.DataFrame([{"Pregnancies": 2, "Glucose": 150, "BloodPressure": 72,
 "SkinThickness": 30, "Insulin": 160, "BMI": 33, "DiabetesPedigreeFunction": .5, "Age": 35}])
print("Example patient's estimated positive-class probability:", f"{model.predict_proba(patient)[0,1]:.1%}")
print("Educational model only; not a medical diagnosis.")
