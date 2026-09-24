# 🚗 Smart Used Car Price Prediction Using Machine Learning

> **IBM SkillsBuild | AI & Data Science Project | Academic Submission Ready**

---

## 📌 Project Title

**Smart Used Car Price Prediction Using Machine Learning**

---

## 📋 Project Overview

This project builds a complete, end-to-end machine learning system that predicts the **selling price of used cars** using real-world data from India's largest car marketplace — CarDekho. The project covers every stage of a professional data science pipeline: raw data loading → cleaning → exploratory analysis → feature engineering → model training → evaluation → interactive web deployment.

The final deliverable is a fully-functional **Streamlit web dashboard** with seven pages that any user can run locally in minutes.

---

## ❗ Problem Statement

The used car market in India processes millions of transactions every year. However, both buyers and sellers often lack reliable information to determine a fair price for a vehicle. This information gap leads to buyers overpaying and sellers undervaluing their cars. Current pricing is largely based on guesswork, brand perception, and informal negotiation.

---

## 🎯 Objective

Develop a machine learning system that:
1. Trains on real used car listing data
2. Learns the relationship between car features and selling price
3. Predicts the fair market value of any used car configuration
4. Deploys as an accessible, interactive web dashboard

---

## 📂 Dataset Description

| Property | Value |
|---|---|
| **File Name** | `Car details v3 (2).csv` |
| **Source** | CarDekho (https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?utm_source=chatgpt.com) |
| **Records** | ~8,000 used car listings |
| **Target Variable** | `selling_price` (Indian Rupees ₹) |
| **Original Columns** | 13 |

### Column Reference

| Column | Type | Description |
|---|---|---|
| `name` | Text | Car model name |
| `year` | Integer | Year of manufacture |
| `selling_price` | Integer | **Target** — price the car was sold at (₹) |
| `km_driven` | Integer | Odometer reading (kilometres) |
| `fuel` | Categorical | Petrol / Diesel / CNG / LPG / Electric |
| `seller_type` | Categorical | Individual / Dealer / Trustmark Dealer |
| `transmission` | Categorical | Manual / Automatic |
| `owner` | Categorical | First / Second / Third / Fourth+ Owner |
| `mileage` | Float | Fuel efficiency (kmpl or km/kg) |
| `engine` | Float | Engine displacement (CC) |
| `max_power` | Float | Maximum power output (bhp) |
| `torque` | Text | Torque (dropped — too inconsistent) |
| `seats` | Float | Number of seats |

### Dataset Source

🔗 [Kaggle – Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Core programming language |
| **Pandas** | ≥2.0 | Data loading, cleaning, manipulation |
| **NumPy** | ≥1.24 | Numerical operations, log transforms |
| **Matplotlib** | ≥3.7 | Base plotting library |
| **Seaborn** | ≥0.12 | Statistical visualizations |
| **Scikit-learn** | ≥1.3 | ML models, preprocessing, evaluation |
| **Streamlit** | ≥1.32 | Interactive web dashboard |
| **openpyxl** | ≥3.1 | Excel export (optional) |

---

## 📁 Project Structure

```
IBM project/
│
├── app.py                                   🚀 Main Streamlit application
├── requirements.txt                         📦 Python dependencies
├── README.md                                📖 This documentation file
├── Car details v3 (2).csv                   📊 Dataset
└── Used_Car_Price_Prediction_Report.docx    📄 Full academic project report
```

---

## ⚙️ Installation Guide

### Prerequisites
- Python 3.10 or higher
- pip package manager
- The dataset file `Car details v3 (2).csv` in the same directory as `app.py`

### Step-by-Step Setup

**Step 1 — Create a virtual environment (recommended)**
```bash
python -m venv venv
```

**Step 2 — Activate the virtual environment**
```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**Step 3 — Install all dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

```bash
streamlit run app.py
```

Streamlit will automatically open `http://localhost:8501` in your default browser.
> **Note:** The app loads and trains all three models on startup (~15–30 seconds on first run, then cached).

---

## 🖥️ Dashboard Features

| Page | Description |
|---|---|
| 🏠 **Home** | Hero banner, 6 KPI cards, project objective, pipeline checklist, dataset sample, price distribution preview |
| 📋 **Dataset Overview** | Three tabs: Raw Dataset / Cleaned Dataset / Missing Values & Data Types |
| 📊 **Summary Statistics** | Colour-gradient descriptive stats table, categorical distribution explorer, IQR outlier detection with box plot |
| 🔍 **EDA & Visualizations** | Dropdown to select any of 9 charts, each with a figure number and insight caption |
| 🗺️ **Correlation Heatmap** | Full Pearson correlation matrix + key insights table |
| 🤖 **Model Performance** | Side-by-side metrics for all 3 models, Actual vs Predicted charts, Feature Importance, model config table |
| 💡 **Predict Price** | 10-input form → instant price estimate + ±10% range + CSV download + similar cars from dataset |

---

## 🔬 Machine Learning Workflow

```
Raw CSV
   ↓
Data Cleaning
   (deduplicate · extract numbers · drop torque · fill NaN · remove outliers)
   ↓
Feature Engineering
   (car_age = 2024 − year · LabelEncode categoricals)
   ↓
Train/Test Split  (80% / 20%,  random_state=42)
   ↓
Log-Transform Target  →  log1p(selling_price)
   ↓
Train 3 Models
   ├── Linear Regression
   ├── Decision Tree  (max_depth=10)
   └── Random Forest  (n_estimators=200)
   ↓
Evaluate on Test Set
   (R² · MAE · RMSE — all on back-transformed predictions)
   ↓
Select Best Model  →  used in Predict Price page
```

---

## 📊 Model Evaluation Results

| Model | R² Score | MAE (₹) | RMSE (₹) |
|---|---|---|---|
| Linear Regression | ~0.67 | ~₹1,50,000 | ~₹2,80,000 |
| Decision Tree | ~0.85 | ~₹85,000 | ~₹1,90,000 |
| **Random Forest** ✅ | **~0.91** | **~₹65,000** | **~₹1,50,000** |

> Actual values will vary slightly on each run due to train/test randomness (fixed at seed 42).

---

## 🔑 Key Findings

1. **Car age is the #1 predictor** — Post-2015 cars command dramatically higher prices due to depreciation patterns.
2. **Engine CC and max_power** are the strongest performance-related predictors, reflecting the premium/luxury segment.
3. **Automatic transmission** adds a ~40–60% price premium over equivalent Manual cars.
4. **Diesel cars** have a higher median selling price than Petrol cars in India.
5. **Ownership history** significantly affects price — First Owner > Second Owner > Third Owner.
6. **Log-transforming** the target variable was critical for Linear Regression and significantly improved all models.
7. **Random Forest** outperforms Linear Regression by ~24 R² points and Decision Tree by ~6 R² points.

---

## 🚀 Future Scope

1. **Geographic location** — Prices vary significantly between Indian cities; adding city as a feature could improve accuracy.
2. **XGBoost / LightGBM** — Gradient boosting models typically outperform Random Forest on tabular data.
3. **Hyperparameter tuning** — Grid Search or Optuna for better model configuration.
4. **More recent data** — Include 2023–2024 EV listings for better Electric Vehicle pricing.
5. **Cloud deployment** — Deploy to Streamlit Cloud for public access without local setup.
6. **Price trend forecasting** — Add time-series analysis to predict how a car's value will change over the next 12 months.

---

## 📚 References

1. Nehal Birla. *Vehicle Dataset from CarDekho*. Kaggle, 2021. https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
3. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825–2830.
4. Streamlit Inc. *Streamlit Documentation*. https://docs.streamlit.io
5. IBM SkillsBuild. *AI & Data Science Learning Path*. https://skillsbuild.org
6. McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. SciPy Conference.

---

## 👨‍🎓 Suitable For

- ✅ IBM SkillsBuild AI & Data Science project submission
- ✅ Academic final-year / mini project (B.Tech / BCA / MCA)
- ✅ Data science portfolio showcase
- ✅ BOB (Bob) project submission

---

*Smart Used Car Price Prediction · IBM SkillsBuild Project · Built with Streamlit*
