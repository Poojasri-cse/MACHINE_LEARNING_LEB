import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from common import load_csv

print("Worksheet 5.8 - Exercise 1: Chicago taxi fare")
url = "https://raw.githubusercontent.com/Amana-Batool/Linear-Regression/main/chicago_taxi_train.csv"
raw = load_csv("chicago_taxi_train.csv", url)
taxi = pd.DataFrame({
    "trip_miles": raw["TRIP_MILES"],
    "trip_seconds": raw["TRIP_SECONDS"],
    "pickup_hour": raw["TRIP_START_HOUR"].astype(int),
    "fare": raw["FARE"]
})
taxi["rush_hour"] = (((taxi.pickup_hour.between(7, 9)) |
                      (taxi.pickup_hour.between(16, 18)))).astype(int)
taxi = taxi.dropna()
X = taxi[["trip_miles", "trip_seconds", "rush_hour"]]
y = taxi["fare"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print("Rows:", len(taxi))
print("Intercept:", round(model.intercept_, 3))
print("Coefficients:", dict(zip(X.columns, model.coef_.round(4))))
print("MSE:", round(mean_squared_error(y_test, pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 4))
print("R2:", round(r2_score(y_test, pred), 4))
new_ride = pd.DataFrame([[6, 22*60, 1]], columns=X.columns)
print("Predicted fare for 6-mile, 22-minute rush-hour trip: $", round(model.predict(new_ride)[0], 2))
plt.scatter(y_test, pred, alpha=.5)
plt.xlabel("Actual fare ($)"); plt.ylabel("Predicted fare ($)")
plt.title("Chicago taxi fare: actual vs predicted"); plt.tight_layout(); plt.show()
