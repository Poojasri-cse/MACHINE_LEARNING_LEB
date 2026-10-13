import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from common import load_csv

print("Worksheet 6.6 - Exercise 1: Graduate admission")
url = "https://raw.githubusercontent.com/divyansha1115/Graduate-Admission-Prediction/main/Admission_Predict_Ver1.1.csv"
df = load_csv("Admission_Predict_Ver1.1.csv", url)
df.columns = df.columns.str.strip()
data = pd.DataFrame({"admit": (df["Chance of Admit"] >= .75).astype(int),
                     "gre": df["GRE Score"], "gpa": df["CGPA"],
                     "rank": df["University Rating"]}).dropna()
X = pd.concat([data[["gre", "gpa"]], pd.get_dummies(data["rank"], prefix="rank", dtype=int)], axis=1)
y = data["admit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
model = LogisticRegression(C=1e4, max_iter=5000).fit(X_train, y_train)
pred, prob = model.predict(X_test), model.predict_proba(X_test)[:, 1]
print("Confusion matrix:\n", metrics.confusion_matrix(y_test, pred))
print("Accuracy:", round(metrics.accuracy_score(y_test, pred), 3))
print("ROC-AUC:", round(metrics.roc_auc_score(y_test, prob), 3))
print(metrics.classification_report(y_test, pred, zero_division=0))
sns.heatmap(metrics.confusion_matrix(y_test, pred), annot=True, fmt="d")
plt.title("Graduate admission confusion matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.show()
def applicant(gre, gpa, rating):
    row = {"gre": gre, "gpa": gpa, **{f"rank_{r}": int(rating == r) for r in range(2, 6)}}
    return pd.DataFrame([row]).reindex(columns=X.columns, fill_value=0)
for rating in (5, 1):
    print(f"Admission probability (GRE 320, GPA 9.0, rating {rating}):",
          f"{model.predict_proba(applicant(320, 9.0, rating))[0, 1]:.1%}")
