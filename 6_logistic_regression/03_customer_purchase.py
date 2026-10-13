import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from common import load_csv

print("Worksheet 6.6 - Exercise 3: Customer purchase prediction")
df = load_csv("Social_Network_Ads.csv",
 "https://raw.githubusercontent.com/sharmaroshan/Social-Networks-Ads/master/Social_Network_Ads.csv")
df = df.drop(columns=["User ID"], errors="ignore").dropna()
X = pd.get_dummies(df[["Gender", "Age", "EstimatedSalary"]], columns=["Gender"], drop_first=True, dtype=int)
y = df["Purchased"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(metrics.accuracy_score(y_test, pred), 3))
print(metrics.classification_report(y_test, pred, zero_division=0))
new = pd.DataFrame([{"Age": 24, "EstimatedSalary": 45000, "Gender_Male": 0}]).reindex(columns=X.columns, fill_value=0)
print("Purchase probability:", f"{model.predict_proba(new)[0,1]:.1%}")
metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Customer purchase confusion matrix"); plt.tight_layout(); plt.show()
