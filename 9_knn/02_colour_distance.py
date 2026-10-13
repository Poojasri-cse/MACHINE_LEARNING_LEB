import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

print("Worksheet 9.6 - Exercise 2: Euclidean distance and KNN")
colours = pd.DataFrame({"Brightness":[40,50,60,10,70,60,25],
 "Saturation":[20,50,90,25,70,10,80],
 "Class":["Red","Blue","Blue","Red","Blue","Red","Blue"]})
new_entry = (20,35)
colours["Distance"] = np.sqrt((colours["Brightness"]-new_entry[0])**2 +
                              (colours["Saturation"]-new_entry[1])**2)
ranked = colours.sort_values("Distance").reset_index(drop=True)
print("All entries ranked by Euclidean distance:")
print(ranked.round(2).to_string(index=False))
k = 5
top = ranked.head(k)
votes = top["Class"].value_counts()
print(f"\n{k} nearest neighbours vote:", votes.to_dict())
print("Predicted class:", votes.idxmax())
model = KNeighborsClassifier(n_neighbors=k).fit(colours[["Brightness","Saturation"]], colours["Class"])
print("scikit-learn KNN prediction:", model.predict([new_entry])[0])
for label, group in colours.groupby("Class"):
    plt.scatter(group.Brightness, group.Saturation, label=label)
plt.scatter(*new_entry, marker="*", s=220, label="New entry")
plt.xlabel("Brightness"); plt.ylabel("Saturation"); plt.title("Colour classification using KNN")
plt.legend(); plt.tight_layout(); plt.show()
