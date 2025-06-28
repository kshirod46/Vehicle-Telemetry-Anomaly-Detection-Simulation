# Vehicle Telemetry Anomaly Detection Simulation

This project demonstrates a step-by-step workflow for simulating vehicle telemetry data, applying various classic anomaly detection models, and comparing their results using Python.  
The workflow is designed for easy use in Google Colab or Jupyter notebooks, with a focus on clarity and extensibility.

---

## Features

- **Synthetic real-time vehicle telemetry data generation** (speed, rpm, throttle, engine temperature, fuel).
- **Multiple classic anomaly detection models**:
  - Isolation Forest
  - One-Class SVM
  - Local Outlier Factor (LOF)
  - K Nearest Neighbors (KNN)
  - Histogram-based Outlier Score (HBOS)
- **Model comparison and visualization**:
  - Number of anomalies detected per model
  - Overlap in flagged anomalies
  - Scatter plots, Venn diagrams, and summary tables

---

## Workflow Overview

### 1. Data Simulation

Generate synthetic vehicle telemetry data with occasional injected anomalies.  
Example features:  
- `speed` (km/h)
- `rpm` (engine revolutions per minute)
- `throttle` (%)
- `engine_temp` (°C)
- `fuel` (%)

### 2. Data Storage

Store the simulated data in a pandas DataFrame (and optionally save as CSV).

### 3. Anomaly Detection

Apply and compare several classic unsupervised anomaly detection algorithms using the [PyOD](https://pyod.readthedocs.io/en/latest/) library:
- Isolation Forest
- One-Class SVM
- LOF
- KNN
- HBOS

Each model outputs a prediction (0 = normal, 1 = anomaly) for each data point.

### 4. Model Comparison & Visualization

- Tabulate and visualize the number of anomalies detected by each model.
- Explore overlap among models (e.g., how many points are flagged by multiple models).
- Visualize anomalies in feature space (scatter plots, Venn diagrams).

---

## Example Code Snippets

### Data Generation

```python
import random
import pandas as pd

def generate_telemetry():
    speed = random.uniform(40, 120)
    rpm = random.uniform(1000, 4000)
    throttle = random.uniform(10, 80)
    engine_temp = random.uniform(70, 110)
    fuel = random.uniform(20, 100)
    if random.randint(1, 20) == 1:
        speed *= random.uniform(1.5, 2)
        rpm *= random.uniform(1.5, 2)
        engine_temp += random.uniform(20, 50)
    return {
        'speed': round(speed, 2),
        'rpm': round(rpm, 2),
        'throttle': round(throttle, 2),
        'engine_temp': round(engine_temp, 2),
        'fuel': round(fuel, 2)
    }

data = [generate_telemetry() for _ in range(100)]
df = pd.DataFrame(data)
```

### Anomaly Detection (Example: Isolation Forest)

```python
from pyod.models.iforest import IForest
features = ['speed', 'rpm', 'throttle', 'engine_temp', 'fuel']
model = IForest()
model.fit(df[features])
df['anomaly'] = model.predict(df[features])
```

### Adding More Models

```python
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS

df['anomaly_ocsvm'] = OCSVM().fit(df[features]).predict(df[features])
df['anomaly_lof'] = LOF().fit(df[features]).predict(df[features])
df['anomaly_knn'] = KNN().fit(df[features]).predict(df[features])
df['anomaly_hbos'] = HBOS().fit(df[features]).predict(df[features])
```

### Visualization

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
plt.plot(df['speed'], label='Speed')
plt.scatter(df.index[df['anomaly'] == 1], df['speed'][df['anomaly'] == 1], color='red', label='IF Anomaly')
plt.scatter(df.index[df['anomaly_knn'] == 1], df['speed'][df['anomaly_knn'] == 1], color='lime', marker='x', label='KNN Anomaly')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Speed (km/h)')
plt.show()
```

---

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- pyod

Install with:
```bash
pip install pandas matplotlib pyod
```

---

## Next Steps

- Apply deep learning models (e.g., AutoEncoder, LSTM) from PyOD.
- Add interactive dashboards (Streamlit, Dash).
- Integrate real-time streaming and automated alerting.

---

## License

MIT License

---

## Acknowledgements

- [PyOD: Python Outlier Detection library](https://pyod.readthedocs.io/en/latest/)
- Example data generation and visualization inspired by open data science tutorials.
