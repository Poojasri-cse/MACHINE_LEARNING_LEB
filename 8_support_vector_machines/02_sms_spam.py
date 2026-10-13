import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import metrics
from common import load_csv

print("Worksheet 8.6 - Exercise 2: SMS spam classification")
sms = load_csv("sms.tsv", "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
               sep="\t", header=None, names=["label", "text"])
sms = sms.drop_duplicates().reset_index(drop=True)
def features(message):
    words = re.findall(r"[a-z']+", message.lower())
    n = max(len(words), 1)
    pct = lambda vocab: 100 * sum(w in vocab for w in words) / n
    letters = [c for c in message if c.isalpha()]
    return {"freq_free": pct({"free"}), "freq_win": pct({"win","won","winner","prize"}),
            "freq_call": pct({"call","txt","mobile","claim"}), "freq_lol": pct({"lol","ok","sure","later"}),
            "message_length": len(message), "num_links": len(re.findall(r"www\.|http", message.lower())),
            "num_digits": sum(c.isdigit() for c in message),
            "capital_ratio": sum(c.isupper() for c in letters)/max(len(letters),1)}
X = pd.DataFrame([features(t) for t in sms["text"]])
y = (sms["label"] == "spam").astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)
pipe = Pipeline([("scale", StandardScaler()), ("svc", SVC(kernel="rbf"))])
grid = GridSearchCV(pipe, {"svc__C":[.1,1,10,100], "svc__gamma":["scale","auto"]}, scoring="f1", cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
pred = grid.predict(X_test)
print("Best parameters:", grid.best_params_)
print("Best CV F1:", round(grid.best_score_, 3))
print("Test accuracy:", round(metrics.accuracy_score(y_test, pred), 3))
print(metrics.classification_report(y_test, pred, target_names=["Not spam","Spam"], zero_division=0))
metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("SMS spam confusion matrix"); plt.tight_layout(); plt.show()
for msg in ["WINNER!! You have won a FREE prize. Call now to claim your reward!",
            "Ok lol, I'll be there later. Sure, see you tomorrow"]:
    print(msg, "=>", "Spam" if grid.predict(pd.DataFrame([features(msg)]))[0] else "Not spam")
