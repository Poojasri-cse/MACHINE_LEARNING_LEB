import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Worksheet 5.8 - Exercise 2: Body weight from height, age and gender")

# Load dataset from the project folder
dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gym_members_exercise_tracking.csv"
)

df = pd.read_csv(dataset_path)

# Standardize column names
df = df.rename(columns={
    "Weight (kg)": "Weight",
    "Height (m)": "Height"
})

# Select required columns and remove missing values
data = df[["Weight", "Height", "Age", "Gender"]].dropna()

# Encode gender into numerical features
X = pd.get_dummies(
    data[["Height", "Age", "Gender"]],
    columns=["Gender"],
    drop_first=True,
    dtype=int
)

y = data["Weight"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test data
pred = model.predict(X_test)

# Evaluate the model
print("\n--- Model Results ---")
print("Dataset shape:", data.shape)
print("Features:", list(X.columns))
print("Intercept:", round(model.intercept_, 3))
print("Coefficients:", dict(zip(X.columns, model.coef_.round(3))))
print("Test R2:", round(r2_score(y_test, pred), 3))
print("Test RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 3))

# Five-fold cross-validation
cv_scores = cross_val_score(
    LinearRegression(), X, y, cv=5, scoring="r2"
)
print("5-fold CV R2:", round(cv_scores.mean(), 3))

# Predict weight for a new person
gender_columns = [c for c in X.columns if c.startswith("Gender_")]

new_person = pd.DataFrame(0, index=[0], columns=X.columns)
new_person["Height"] = 1.75
new_person["Age"] = 30

# Match the dataset's gender encoding
if gender_columns:
    male_column = next(
        (c for c in gender_columns if c.split("_", 1)[1].lower() == "male"),
        None
    )
    if male_column:
        new_person[male_column] = 1

predicted_weight = model.predict(new_person)[0]

print(
    "Predicted weight for height 1.75 m, age 30, male:",
    round(predicted_weight, 2), "kg"
)

# Plot actual vs predicted weights
plt.figure(figsize=(8, 5))
plt.scatter(y_test, pred, alpha=0.6)
plt.xlabel("Actual weight (kg)")
plt.ylabel("Predicted weight (kg)")
plt.title("Body Weight Regression")
plt.tight_layout()
plt.show()
