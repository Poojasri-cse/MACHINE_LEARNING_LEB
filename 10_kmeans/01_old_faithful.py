import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("Worksheet 10.6 - Exercise 1: Old Faithful Geyser Clustering")

dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "OldFaithful.csv"
)

df = pd.read_csv(dataset_path)

df = df.rename(columns={
    "TimeEruption": "duration",
    "TimeWaiting": "waiting"
}).dropna()

X = df[["waiting", "duration"]]

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

wcss = []
sil = {}

for k in range(1, 9):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(Xs)
    wcss.append(model.inertia_)

    if k > 1:
        sil[k] = silhouette_score(Xs, labels)

print("Silhouette scores:", {k: round(v, 3) for k, v in sil.items()})

model = KMeans(n_clusters=2, n_init=10, random_state=42)
df["cluster"] = model.fit_predict(Xs)

centers = pd.DataFrame(
    scaler.inverse_transform(model.cluster_centers_),
    columns=X.columns
)

print("Cluster centers:\n", centers.round(2).to_string(index=False))
print("Cluster sizes:\n", df["cluster"].value_counts().sort_index().to_string())
print("Final silhouette:", round(silhouette_score(Xs, df["cluster"]), 3))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(range(1, 9), wcss, "o-")
ax[0].set(
    title="Elbow Method",
    xlabel="Number of Clusters",
    ylabel="WCSS"
)

for cluster, group in df.groupby("cluster"):
    ax[1].scatter(
        group["waiting"],
        group["duration"],
        label=f"Cluster {cluster}"
    )

ax[1].scatter(
    centers["waiting"],
    centers["duration"],
    marker="X",
    s=180,
    c="black",
    label="Centers"
)

ax[1].set(
    xlabel="Waiting (minutes)",
    ylabel="Duration (minutes)",
    title="Old Faithful Geyser Clusters"
)
ax[1].legend()

plt.tight_layout()
plt.show()
