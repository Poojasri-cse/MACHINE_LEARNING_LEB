import re
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import metrics

print("Worksheet 8.6 - Exercise 2: SMS Spam Classification")

# Load dataset
dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sms.tsv"
)

sms = pd.read_csv(
    dataset_path,
    sep="\t",
    header=None,
    names=["label", "text"]
)

sms = sms.drop_duplicates().reset_index(drop=True)

print("Dataset shape:", sms.shape)
print(sms["label"].value_counts())

# Extract numerical features from each message
def features(message):
    words = re.findall(r"[a-z']+", str(message).lower())
    n = max(len(words), 1)

    def pct(vocab):
        return 100 * sum(w in vocab for w in words) / n

    letters = [c for c in str(message) if c.isalpha()]

    return {
        "freq_free": pct({"free"}),
        "freq_win": pct({"win", "won", "winner", "prize"}),
        "freq_call": pct({"call", "txt", "mobile", "claim"}),
        "freq_lol": pct({"lol", "ok", "sure", "later"}),
        "message_length": len(str(message)),
        "num_links": len(re.findall(
            r"(?:https?://|www\.)\S+",
            str(message).lower()
        )),
        "num_digits": sum(c.isdigit() for c in str(message)),
        "capital_ratio": (
            sum(c.isupper() for c in letters) / max(len(letters), 1)
        )
    }

X = pd.DataFrame([features(t) for t in sms["text"]])
y = (sms["label"] == "spam").astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Create SVM pipeline
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("svc", SVC(kernel="rbf"))
])

# Find the best hyperparameters
grid = GridSearchCV(
    pipe,
    {
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", "auto"]
    },
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

# Evaluate model
pred = grid.predict(X_test)

print("\nBest parameters:", grid.best_params_)
print("Best CV F1:", round(grid.best_score_, 3))
print("Test accuracy:", round(metrics.accuracy_score(y_test, pred), 3))
print("Test F1:", round(metrics.f1_score(y_test, pred), 3))

print("\nClassification report:")
print(metrics.classification_report(
    y_test,
    pred,
    target_names=["Not spam", "Spam"],
    zero_division=0
))

# Display confusion matrix
metrics.ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred,
    display_labels=["Not spam", "Spam"],
    cmap="Blues"
)

plt.title("SMS Spam Confusion Matrix")
plt.tight_layout()
plt.show()

# Test new messages
messages = [
    "WINNER!! You have won a FREE prize. Call now to claim your reward!",
    "Ok lol, I'll be there later. Sure, see you tomorrow"
]

for msg in messages:
    row = pd.DataFrame([features(msg)], columns=X.columns)
    result = grid.predict(row)[0]
    print(f"\nMessage: {msg}")
    print("Prediction:", "Spam" if result == 1 else "Not spam")
