import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

print("Worksheet 7.6 - Exercise 2: Go out or stay in based on weather")
data = pd.DataFrame({
 "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
 "Temperature": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
 "Humidity": ["High","High","High","High","Normal","Normal","Normal","High","Normal","Normal","Normal","High","Normal","High"],
 "Wind": ["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
 "Decision": ["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"]})
data["Decision"] = data["Decision"].map({"Yes":"Go out", "No":"Stay in"})
X = pd.get_dummies(data.drop(columns="Decision"), dtype=int)
y = data["Decision"]
model = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
print("Training accuracy:", round(model.score(X, y), 3))
print(export_text(model, feature_names=list(X.columns)))
new_days = pd.DataFrame([
 {"Outlook":"Sunny","Temperature":"Mild","Humidity":"Normal","Wind":"Weak"},
 {"Outlook":"Rain","Temperature":"Hot","Humidity":"High","Wind":"Weak"},
 {"Outlook":"Overcast","Temperature":"Cool","Humidity":"High","Wind":"Strong"}])
new_X = pd.get_dummies(new_days).reindex(columns=X.columns, fill_value=0)
print(pd.DataFrame({"Weather": ["Sunny/Mild/Normal/Weak","Rain/Hot/High/Weak","Overcast/Cool/High/Strong"],
                    "Prediction": model.predict(new_X)}).to_string(index=False))
plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=list(X.columns), class_names=list(model.classes_), filled=True)
plt.title("Weather decision tree"); plt.tight_layout(); plt.show()
