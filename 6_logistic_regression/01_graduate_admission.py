import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from common import load_csv

print("Worksheet 6.6 - Exercise 1: Graduate Admission")

dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Admission_Predict_Ver1.1.csv"
)

df = pd.read_csv(dataset_path)
df.columns = df.columns.str.strip()

required = [
    "Chance of Admit", "GRE Score", "CGPA", "University Rating"
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns: {missing}\nAvailable columns: {df.columns.tolist()}"
    )

data = pd.DataFrame({
    "admit": (df["Chance of Admit"] >= 0.75).astype(int),
    "gre": df["GRE Score"],
    "gpa": df["CGPA"],
    "rank": df["University Rating"]
}).dropna()

X = pd.concat([
    data[["gre", "gpa"]],
    pd.get_dummies(data["rank"], prefix="rank", dtype=int)
], axis=1)

y = data["admit"]

if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError(
        "The admission threshold produced too few examples in one class."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(C=1e4, max_iter=5000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("\nConfusion matrix:\n", metrics.confusion_matrix(y_test, pred))
print("Accuracy:", round(metrics.accuracy_score(y_test, pred), 3))
print("ROC-AUC:", round(metrics.roc_auc_score(y_test, prob), 3))
print(metrics.classification_report(y_test, pred, zero_division=0))

cm = metrics.confusion_matrix(y_test, pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Graduate Admission Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

def applicant(gre, gpa, rating):
    row = {
        "gre": gre,
        "gpa": gpa,
        **{
            col: int(col == f"rank_{rating}")
            for col in X.columns if col.startswith("rank_")
        }
    }
    return pd.DataFrame([row]).reindex(columns=X.columns, fill_value=0)

for rating in (5, 1):
    probability = model.predict_proba(
        applicant(320, 9.0, rating)
    )[0, 1]

    print(
        f"Admission probability (GRE 320, CGPA 9.0, "
        f"rating {rating}): {probability:.1%}"
    )
