import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

print("Worksheet 11.4 - Exercise 1: House Price PCA")

dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "kc_house_data.csv"
)

df = pd.read_csv(dataset_path)

features = [
    "sqft_living", "sqft_lot", "bedrooms", "bathrooms", "floors",
    "waterfront", "view", "condition", "grade", "age_years",
    "sqft_basement", "sqft_living15"
]

props = pd.DataFrame({
    "price_k": df["price"] / 1000,
    "sqft_living": df["sqft_living"],
    "sqft_lot": df["sqft_lot"],
    "bedrooms": df["bedrooms"],
    "bathrooms": df["bathrooms"],
    "floors": df["floors"].astype(float),
    "waterfront": df["waterfront"],
    "view": df["view"],
    "condition": df["condition"],
    "grade": df["grade"],
    "age_years": 2015 - df["yr_built"],
    "sqft_basement": df["sqft_basement"],
    "sqft_living15": df["sqft_living15"]
})

props = props.dropna()
props["high_price"] = (
    props["price_k"] > props["price_k"].median()
).astype(int)

print("Rows:", len(props))
print("Median sale price ($):", round(props["price_k"].median() * 1000))

comparison = props.groupby("high_price")[features].mean().T
comparison.columns = ["Lower-priced", "High-priced"]

print("\nGroup comparison:\n", comparison.round(2).to_string())

scaler = StandardScaler()
Xs = scaler.fit_transform(props[features])

pca = PCA()
scores = pca.fit_transform(Xs)

evr = pca.explained_variance_ratio_

print("\nExplained variance:", evr.round(3))
print("Cumulative variance:", evr.cumsum().round(3))

loadings = pd.DataFrame(
    pca.components_[:3].T,
    index=features,
    columns=["PC1", "PC2", "PC3"]
)

print("\nFirst three component loadings:\n", loadings.round(3).to_string())

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(range(1, len(evr) + 1), evr, "o-")
ax[0].set(
    title="Scree Plot",
    xlabel="Component",
    ylabel="Variance Ratio"
)

ax[1].scatter(
    scores[:, 0], scores[:, 1],
    c=props["high_price"], s=8, alpha=0.5
)
ax[1].set(
    title="Properties in PC1-PC2 Space",
    xlabel="PC1",
    ylabel="PC2"
)

plt.tight_layout()
plt.show()

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=0,
    n_jobs=-1
)

cv_scores = cross_val_score(
    rf, props[features], props["high_price"], cv=5
)

print("5-fold RF accuracy:", round(cv_scores.mean(), 3))

rf.fit(props[features], props["high_price"])

importance = pd.Series(
    rf.feature_importances_,
    index=features
).sort_values(ascending=False)

print("Feature importance:\n", importance.round(3).to_string())

importance.sort_values().plot.barh(
    title="Random Forest Feature Importance",
    figsize=(7, 4)
)

plt.tight_layout()
plt.show()
