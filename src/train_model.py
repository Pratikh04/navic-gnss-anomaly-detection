import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import os

df = pd.read_csv("data/navic_sample.csv")

features = [
"CNo",
"Elevation",
"Azimuth",
"Doppler",
"LockTime",
"RangeResidual",
"SatCount"
]

X = df[features]

X = X.fillna(X.median())

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = IsolationForest(
n_estimators=200,
contamination=0.02,
random_state=42
)

model.fit(X_scaled)

df["anomaly"] = model.predict(X_scaled)

df["anomaly"] = df["anomaly"].map({1:0,-1:1})

os.makedirs("outputs",exist_ok=True)

df[df["anomaly"]==1].to_csv(
"outputs/navic_anomalies.csv",
index=False
)

print("training finished")
