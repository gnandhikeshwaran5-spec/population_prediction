# 🇮🇳 India Population Intelligence — Streamlit App

A production-grade ML forecasting dashboard for India's population dynamics (1950–2075).

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run india_population_forecast_app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Push both files to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your repo → select `india_population_forecast_app.py`
4. Click **Deploy** — done!

## ✨ Features

| Feature | Description |
|---|---|
| 📈 **Live Forecast** | Real-time recomputation when you adjust any sidebar control |
| 🎚️ **Train/Test Split** | Drag slider to change training cutoff year |
| 🔭 **Multi-Scenario** | Low / Baseline / High growth projections with confidence bands |
| 🤖 **5 ML Models** | Linear, Poly-2, Poly-3, Random Forest, Gradient Boosting |
| 📊 **Residual Diagnostics** | Full residual plots, histograms, actual-vs-predicted |
| 🌐 **Growth Dynamics** | Demographic transition visualization with acceleration |
| 🔬 **What-If Simulator** | Define custom growth trajectories + demographic shocks |
| 🌍 **UN Comparison** | Overlay official UN medium-variant estimates |

## 📁 File Structure

```
├── india_population_forecast_app.py   # Main Streamlit app
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```
