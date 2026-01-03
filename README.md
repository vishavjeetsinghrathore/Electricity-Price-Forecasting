# ⚡ Electricity Load Forecasting using Machine Learning

## 📌 Project Overview
This project predicts **hourly electricity load (AEP_MW)** using:
- Time-series machine learning
- Sliding window technique
- TimeSeriesSplit cross-validation
- Random Forest regression
- Streamlit dashboard

---

## 📊 Dataset
Source: Kaggle – PJM Hourly Energy Consumption  
Columns:
- `Datetime` → Timestamp
- `AEP_MW` → Electricity load (MW)

---

## 🧠 ML Concepts Used
- Lag features (sliding window)
- Time-based features (hour, day, month)
- Time-series cross-validation
- Hyperparameter tuning
- Residual-aware evaluation

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

### Project live on

https://electricity-price-forecasting-gelgu9mjrdeylewmysefvd.streamlit.app/