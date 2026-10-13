import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

print("Worksheet 6.6 - Exercise 2: Diabetes Risk Classification")

# Load dataset from project folder
dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "diabetes.csv"
)

df = pd.read_csv(dataset_path)
df.columns = df.columns.str.strip()

# Standardize target column name
df = df.rename(columns={"Outcome": "disease"})

required = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "disease"
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns: {missing}\nAvailable columns: {df.columns.tolist()}"
    )

df = df[required].dropna()

X = df.drop(columns="disease")
y = df["disease"].astype(int)

# Replace medically implausible zero measurements with missing values
zero_missing = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]
X[zero_missing] = X[zero_missing].replace(0, np.nan)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build and train the model
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

model.fit(X_train, y_train)

# Make predictions
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nDataset shape:", df.shape)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nConfusion matrix:\n", metrics.confusion_matrix(y_test, pred))
print("Accuracy:", round(metrics.accuracy_score(y_test, pred), 3))
print("Precision:", round(metrics.precision_score(y_test, pred, zero_division=0), 3))
print("Recall:", round(metrics.recall_score(y_test, pred, zero_division=0), 3))
print("F1-score:", round(metrics.f1_score(y_test, pred, zero_division=0), 3))
print("ROC-AUC:", round(metrics.roc_auc_score(y_test, prob), 3))

print("\nClassification report:")
print(metrics.classification_report(
    y_test, pred,
    target_names=["No diabetes", "Diabetes"],
    zero_division=0
))

# Plot confusion matrix
metrics.ConfusionMatrixDisplay.from_predictions(
    y_test, pred,
    display_labels=["No diabetes", "Diabetes"],
    cmap="Blues"
)

plt.title("Diabetes Risk Classification")
plt.tight_layout()
plt.show()

# Predict for an example patient
patient = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 150,
    "BloodPressure": 72,
    "SkinThickness": 30,
    "Insulin": 160,
    "BMI": 33,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 35
}])

probability = model.predict_proba(patient)[0, 1]

print(
    "\nExample patient's estimated positive-class probability:",
    f"{probability:.1%}"
)

print("Educational model only; not a medical diagnosis.")
