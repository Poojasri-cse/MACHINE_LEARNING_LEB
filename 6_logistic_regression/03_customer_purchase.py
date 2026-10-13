import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

print("Worksheet 6.6 - Exercise 3: Customer Purchase Prediction")

dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Social_Network_Ads.csv"
)

df = pd.read_csv(dataset_path)
df = df.drop(columns=["User ID"], errors="ignore").dropna()

X = pd.get_dummies(
    df[["Gender", "Age", "EstimatedSalary"]],
    columns=["Gender"],
    drop_first=True,
    dtype=int
)

y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", round(metrics.accuracy_score(y_test, pred), 3))
print("\nClassification Report:\n")
print(metrics.classification_report(y_test, pred, zero_division=0))

new_customer = pd.DataFrame([
    {"Age": 24, "EstimatedSalary": 45000, "Gender_Male": 0}
]).reindex(columns=X.columns, fill_value=0)

probability = model.predict_proba(new_customer)[0, 1]

print("New Customer Details: Age = 24, Salary = 45000")
print("Purchase Probability:", f"{probability:.1%}")

metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Customer Purchase Confusion Matrix")
plt.tight_layout()
plt.show()
