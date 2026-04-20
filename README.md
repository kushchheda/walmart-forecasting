# Walmart Sales Forecasting

![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Framework](https://img.shields.io/badge/Framework-Statsmodels%20%7C%20Prophet-orange)

Walmart weekly sales time series forecasting pipeline built on 3 years of weekly sales data
across 45 Walmart stores. Four models are compared: Moving Average baseline, ARIMA, SARIMA,
and Prophet, evaluated on a held-out 13-week test set.

---

## 📌 Executive Summary (STAR Method)

* **Situation:** Retailers face significant financial risks from inventory imbalances. Overstocking leads to high holding costs, while understocking during peak periods like Black Friday results in lost revenue and diminished loyalty.
* **Task:** Develop a robust forecasting system using 143 weeks of data across 45 Walmart stores to predict demand spikes and quantify the impact of holidays and macroeconomic factors (CPI, Unemployment, Fuel).
* **Action:** Built a modular Python production pipeline to compare four modeling philosophies: **Moving Average (Baseline)**, **ARIMA**, **SARIMA**, and **Meta’s Prophet**. Implemented custom preprocessing for date-alignment, stationarity testing (ADF), and seasonal decomposition.
* **Result:** Achieved a **23.9% improvement** in accuracy over the baseline. The winning model, **Prophet**, delivered a **MAPE of 2.16%**, providing a highly reliable engine for inventory procurement.

---

## 🚀 Key Results

| Model | MAPE (%) | Improvement vs Baseline |
| :--- | :--- | :--- |
| **Prophet** | **2.16%** | 23.9% |
| Baseline MA(4) | 2.84% | 0% |
| ARIMA(5,1,0) | 3.18% | -12.0% |
| SARIMA(3,1,3)x(0,1,2,13) | 9.69% | -241.2% |

**Prophet achieves a ~24% MAPE improvement over the moving average baseline.**

---

## Project Structure

```
walmart-forecasting/
│
├── data/
│   └── Walmart.csv              ← raw data (not committed to Git)
│
├── src/
│   ├── clean.py                 ← loading, validation, aggregation
│   ├── eda.py                   ← exploratory analysis plots
│   ├── features.py              ← moving average, train/test split
│   ├── models.py                ← ARIMA, SARIMA, Prophet
│   └── evaluate.py              ← metrics, comparison, output plots
│
├── outputs/
│   ├── plots/                   ← all charts (PNG)
│   └── results/                 ← forecasts.csv, model_comparison.csv
│
├── config.py                    ← all constants and parameters
├── main.py                      ← pipeline entry point
├── requirements.txt
└── README.md
```

---

## Setup and Usage

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/walmart-forecasting.git
cd walmart-forecasting
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the data**

Download the [Walmart Sales Dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset)
from Kaggle and place it at `data/Walmart.csv`.

**4. Run the pipeline**
```bash
python main.py
```

All plots will be saved to `outputs/plots/` and all result CSVs to `outputs/results/`.

---

## Dataset

- **Source:** Kaggle - Walmart Store Sales Forecasting
- **Coverage:** 45 stores, 143 weeks (Feb 2010 – Oct 2012)
- **Records:** 6,435 rows (store × week)
- **Features:** Weekly Sales, Holiday Flag, Temperature, Fuel Price, CPI, Unemployment

---

## Methodology

### Data Preparation
- Parsed dates from `%d-%m-%Y` format
- Removed negative sales entries (known data quality issue)
- Aggregated all 45 stores into a single weekly total for national-level forecasting
- Exogenous features averaged weekly; Holiday Flag takes weekly max

### Stationarity
The Augmented Dickey-Fuller test confirms the aggregated series is stationary
(ADF = -5.91, p < 0.0001), making it suitable for classical time series models
without additional differencing.

### Models

**Baseline - Moving Average (4-week)**
A flat forecast using the trailing 4-week average. A `shift(1)` is applied before
rolling to prevent data leakage.

**ARIMA(5,1,0)**
Order selected from ACF/PACF analysis. `p=5` captures autoregressive structure;
`d=1` handles the mild trend; `q=0` showed no significant MA terms.

**SARIMA(3,1,3)x(0,1,2,13)**
Parameters selected via AIC-minimizing grid search over 288 combinations.
Seasonal period `m=13` reflects quarterly patterns in weekly retail data.

**Prophet**
Meta's forecasting library with multiplicative seasonality to handle the scaling
of holiday spikes. Weekly and yearly seasonality enabled; daily disabled (not
applicable to weekly data).

### Evaluation
All models evaluated on the final 13 weeks of data (held out before any model
was trained). Metrics: MAE, RMSE, MAPE.

---

## Key Findings

1. **Prophet is the strongest model** (MAPE 2.16%), outperforming all others by capturing
   multiplicative holiday seasonality effectively.
2. **Holiday weeks boost sales by ~7.8%** - the single most influential external factor.
3. **Economic indicators (CPI, Fuel Price, Temperature, Unemployment) explain less than 3%
   of variance**, suggesting that calendar and seasonal effects dominate.
4. **SARIMA underperforms** on this dataset due to difficulty modeling extreme holiday spikes
   within a 13-week seasonal period.
5. **Future directions:** store-level hierarchical models, XGBoost with lag features, LSTM,
   and Temporal Fusion Transformer for further accuracy gains.

---

## Tech Stack

- **Python 3.9+**
- pandas, numpy, scipy
- statsmodels (ARIMA, SARIMA, ADF test, decomposition)
- prophet (Meta's forecasting library)
- scikit-learn (regression, metrics)
- matplotlib, seaborn
