import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

print("Worksheet 9.6 - Exercise 1: Handwritten digit recognition using KNN")
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)
ks = range(1,16)
scores = []
for k in ks:
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
best_k = list(ks)[int(np.argmax(scores))]
model = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy per K:", dict(zip(ks, np.round(scores,4))))
print("Best K:", best_k)
print("Test accuracy:", round(metrics.accuracy_score(y_test,pred),4))
print(metrics.classification_report(y_test,pred,zero_division=0))
fig, ax = plt.subplots(1,2,figsize=(11,4))
ax[0].plot(list(ks),scores,"o-"); ax[0].set(xlabel="K",ylabel="Test accuracy",title="Choose K")
sns.heatmap(metrics.confusion_matrix(y_test,pred),annot=False,cmap="Blues",ax=ax[1])
ax[1].set(title="Confusion matrix",xlabel="Predicted",ylabel="Actual")
plt.tight_layout(); plt.show()
