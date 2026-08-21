# Anomaly Detection with LSTM Autoencoder

Phat hien hanh vi bat thuong tren log he thong (Sysmon) bang LSTM Autoencoder va giai thich bang Integrated Gradients.

## Cau truc du an

```
DoAn/
|-- pyproject.toml           # Package config (pip install -e .)
|-- requirements.txt         # Thu vien can thiet
|-- data/                    # Du lieu CSV (Normal.csv, NormalVsMalicious01.csv)
|-- src/
|   +-- anomaly_detection/   # Core library
|       |-- config.py        # Cau hinh tap trung (HyperParams + Paths dataclasses)
|       |-- model.py         # LSTM Autoencoder (Keras)
|       |-- dataset.py       # Tao sequences, tinh nhan chuoi
|       |-- preprocess.py    # Ma hoa dac trung + chuan hoa (Preprocessor class)
|       |-- trainer.py       # Pipeline huan luyen
|       |-- evaluator.py     # Pipeline danh gia + metrics
|       |-- detector.py      # Pipeline phat hien bat thuong
|       |-- explainer.py     # Integrated Gradients + tao dashboard data
|       +-- visualization.py # Ve bieu do loss / phan phoi sai so
|-- scripts/                 # Entry-point scripts (ngan gon, chi goi core library)
|   |-- train.py             # Huan luyen mo hinh
|   |-- evaluate.py          # Danh gia mo hinh
|   |-- detect_anomalies.py  # Phat hien bat thuong
|   |-- explain.py           # Giai thich bang Integrated Gradients
|   +-- generate_dashboard.py# Tao du lieu cho web dashboard
+-- artifacts/               # Ket qua
    |-- models/              # Mo hinh .keras, scaler.pkl, encoders.pkl
    +-- results/             # CSV, JSON, PNG, JS
```

## Cai dat

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\pip install -e .
```

## Su dung

```powershell
# 1. Huan luyen
venv\Scripts\python scripts\train.py --epochs 20

# 2. Danh gia
venv\Scripts\python scripts\evaluate.py

# 3. Phat hien bat thuong
venv\Scripts\python scripts\detect_anomalies.py

# 4. Giai thich (top 10 bat thuong nang nhat)
venv\Scripts\python scripts\explain.py --limit 10

# 5. Tao du lieu dashboard
venv\Scripts\python scripts\generate_dashboard.py
```

## Ket qua

| Buoc | Ket qua |
|------|---------|
| ROC AUC | 0.9382 |
| Nguong phat hien | p95 cua sai so tren du lieu binh thuong |
| Nhan chuoi | Bat thuong neu co bat ky su kien nao trong chuoi co Label != 0 |
| Giai thich | Integrated Gradients (backend-agnostic: PyTorch / TensorFlow / JAX) |