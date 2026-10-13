import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn import metrics

print("Worksheet 7.6 - Exercise 1: House Purchase Classification")

dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "housing.csv"
)

df = pd.read_csv(dataset_path)

data = pd.DataFrame({
    "age": df["housing_median_age"],
    "income": df["median_income"] * 10000,
    "bought_house": (
        df["median_house_value"] > df["median_house_value"].median()
    ).astype(int)
}).dropna()

X = data[["age", "income"]]
y = data["bought_house"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=2,
    random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, pred))
print("Accuracy:", round(metrics.accuracy_score(y_test, pred), 3))

print("\nDecision Tree Rules:")
print(export_text(model, feature_names=list(X.columns)))

print("\nFeature Importance:")
print(dict(zip(X.columns, model.feature_importances_.round(3))))

plt.figure(figsize=(10, 5))
plot_tree(
    model,
    feature_names=list(X.columns),
    class_names=["Below median", "Above median"],
    filled=True,
    rounded=True
)
plt.title("House Value Decision Tree")
plt.tight_layout()
plt.show()
