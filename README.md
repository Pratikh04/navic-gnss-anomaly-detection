GNSS / NavIC Signal Anomaly Detection

This project detects anomalies in GNSS/NavIC satellite signal measurements using Isolation Forest.

Features used:

CNo
Elevation
Azimuth
Doppler
LockTime
RangeResidual
SatCount

Run:

pip install -r requirements.txt

python src/train_model.py

Outputs are saved in outputs/navic_anomalies.csv