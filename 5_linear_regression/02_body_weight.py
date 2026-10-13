import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from common import load_csv

print("Worksheet 5.8 - Exercise 2: Body weight from height, age and gender")
url = "https://raw.githubusercontent.com/Mohammadprsd/GYM-members-exercise-tracking-dataset/main/gym_members_exercise_tracking.csv"
df = load_csv("gym_members_exercise_tracking.csv", url)
df = df.rename(columns={"Weight (kg)": "Weight", "Height (m)": "Height"})
data = df[["Weight", "Height", "Age", "Gender"]].dropna()
X = pd.get_dummies(data[["Height", "Age", "Gender"]], columns=["Gender"], drop_first=True, dtype=int)
y = data["Weight"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print("Features:", list(X.columns))
print("Intercept:", round(model.intercept_, 3))
print("Coefficients:", dict(zip(X.columns, model.coef_.round(3))))
print("Test R2:", round(r2_score(y_test, pred), 3))
print("Test RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 3))
print("5-fold CV R2:", round(cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2").mean(), 3))
new = pd.DataFrame([[1.75, 30, 1]], columns=X.columns)
print("Predicted weight for 1.75 m, age 30, male:", round(model.predict(new)[0], 2), "kg")
plt.scatter(y_test, pred, alpha=.6)
plt.xlabel("Actual weight (kg)"); plt.ylabel("Predicted weight (kg)")
plt.title("Body weight regression"); plt.tight_layout(); plt.show()
