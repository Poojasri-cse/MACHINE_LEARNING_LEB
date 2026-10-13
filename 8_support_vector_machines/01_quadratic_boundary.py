import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

print("Worksheet 8.6 - Exercise 1: SVM on quadratic data")
rng = np.random.default_rng(0)
n = 500
x1, x2 = rng.uniform(-3, 3, n), rng.uniform(-2, 10, n)
y = (x2 > x1**2 + rng.normal(0, .5, n)).astype(int)
X = np.column_stack([x1, x2])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)
models = {"Linear": SVC(kernel="linear"), "Polynomial degree 2": SVC(kernel="poly", degree=2, C=10, coef0=1),
          "RBF": SVC(kernel="rbf", C=10, gamma="scale")}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: train={model.score(X_train,y_train):.3f}, test={model.score(X_test,y_test):.3f}, support vectors={len(model.support_vectors_)}")
xx, yy = np.meshgrid(np.linspace(-3,3,200), np.linspace(-2,10,200))
fig, axes = plt.subplots(1, 3, figsize=(14,4), sharey=True)
for ax, (name, model) in zip(axes, models.items()):
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=.25); ax.scatter(X_test[:,0], X_test[:,1], c=y_test, s=10)
    ax.set_title(f"{name}: {model.score(X_test,y_test):.2f}"); ax.set_xlabel("x1")
axes[0].set_ylabel("x2"); plt.tight_layout(); plt.show()
print(metrics.classification_report(y_test, models["RBF"].predict(X_test), zero_division=0))
