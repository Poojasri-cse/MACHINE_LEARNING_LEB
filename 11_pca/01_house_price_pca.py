import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from common import load_csv

print("Worksheet 11.4 - Exercise 1: House price PCA")
url = "https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv"
df = load_csv("kc_house_data.csv", url)
features = ["sqft_living","sqft_lot","bedrooms","bathrooms","floors","waterfront",
            "view","condition","grade","age_years","sqft_basement","sqft_living15"]
props = pd.DataFrame({
 "price_k":df["price"]/1000, "sqft_living":df["sqft_living"], "sqft_lot":df["sqft_lot"],
 "bedrooms":df["bedrooms"], "bathrooms":df["bathrooms"], "floors":df["floors"].astype(float),
 "waterfront":df["waterfront"], "view":df["view"], "condition":df["condition"], "grade":df["grade"],
 "age_years":2015-df["yr_built"], "sqft_basement":df["sqft_basement"], "sqft_living15":df["sqft_living15"]})
props["high_price"] = (props.price_k > props.price_k.median()).astype(int)
props = props.dropna()
print("Rows:",len(props),"Median sale price ($):",round(props.price_k.median()*1000))
comparison = props.groupby("high_price")[features].mean().T
comparison.columns = ["Lower-priced","High-priced"]
print("\nGroup comparison:\n",comparison.round(2).to_string())
Xs = StandardScaler().fit_transform(props[features])
pca = PCA().fit(Xs)
evr = pca.explained_variance_ratio_
print("\nExplained variance:",evr.round(3))
print("Cumulative variance:",evr.cumsum().round(3))
loadings = pd.DataFrame(pca.components_[:3].T,index=features,columns=["PC1","PC2","PC3"])
print("\nFirst three component loadings:\n",loadings.round(3).to_string())
scores = pca.transform(Xs)
fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].plot(range(1,len(evr)+1),evr,"o-"); ax[0].set(title="Scree plot",xlabel="Component",ylabel="Variance ratio")
ax[1].scatter(scores[:,0],scores[:,1],c=props.high_price,s=8,alpha=.5)
ax[1].set(title="Properties in PC1-PC2 space",xlabel="PC1",ylabel="PC2")
plt.tight_layout(); plt.show()
rf=RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=-1)
print("5-fold RF accuracy:",round(cross_val_score(rf,props[features],props.high_price,cv=5).mean(),3))
rf.fit(props[features],props.high_price)
importance=pd.Series(rf.feature_importances_,index=features).sort_values()
print("Feature importance:\n",importance.sort_values(ascending=False).round(3).to_string())
importance.plot.barh(title="Random forest feature importance",figsize=(7,4))
plt.tight_layout(); plt.show()
