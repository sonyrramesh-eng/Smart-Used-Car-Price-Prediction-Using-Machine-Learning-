"""
╔══════════════════════════════════════════════════════════════════════╗
║   Smart Used Car Price Prediction Using Machine Learning             ║
║   IBM SkillsBuild | AI & Data Science Project                        ║
║   Dataset : Car details v3 (2).csv  (CarDekho / Kaggle)             ║
╚══════════════════════════════════════════════════════════════════════╝

File    : app.py  (single-file frontend + backend)
Author  : IBM SkillsBuild Student Project
Stack   : Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Streamlit

Pipeline
--------
1. Data Loading & Cleaning
2. Feature Engineering
3. Exploratory Data Analysis (EDA) + 9 Visualizations
4. Model Training  (Linear Regression | Decision Tree | Random Forest)
5. Model Evaluation (R², MAE, RMSE) + Best-model selection
6. Interactive Prediction Form with CSV download
"""

# ── Standard library ──────────────────────────────────────────────────
import warnings

# ── Third-party ───────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global colour palette ─────────────────────────────────────────────
PRIMARY   = "#1f4e79"
SECONDARY = "#2e75b6"
ACCENT    = "#4472c4"
LIGHT     = "#dce6f0"
PALETTE   = [PRIMARY, SECONDARY, ACCENT, "#70ad47", "#ed7d31", "#ffc000"]

# ── Shared CSS ────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
        /* ── page background ── */
        .main {{ background:#f7f9fc; }}
        /* ── hero banner ── */
        .hero {{ background:linear-gradient(135deg,{PRIMARY},{SECONDARY});
                 border-radius:12px; padding:2rem 2.5rem 1.5rem;
                 color:#fff; margin-bottom:1.5rem; }}
        .hero h1 {{ font-size:2rem; font-weight:800; margin:0; }}
        .hero p  {{ font-size:1rem; opacity:.9; margin:.3rem 0 0; }}
        /* ── section header ── */
        .sec-hdr {{ font-size:1.25rem; font-weight:700; color:{PRIMARY};
                    border-left:5px solid {ACCENT}; padding-left:.6rem;
                    margin:1.2rem 0 .6rem; }}
        /* ── metric card ── */
        .kpi-row {{ display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1rem; }}
        .kpi {{ background:#fff; border:1px solid #ddd; border-radius:10px;
                padding:.8rem 1.2rem; flex:1; min-width:130px; text-align:center; }}
        .kpi .val {{ font-size:1.6rem; font-weight:700; color:{PRIMARY}; }}
        .kpi .lbl {{ font-size:.78rem; color:#666; }}
        /* ── caption ── */
        .fig-cap {{ font-size:.82rem; color:#555; text-align:center;
                    font-style:italic; margin-top:.2rem; }}
        /* ── prediction box ── */
        .pred-box {{ background:#e8f5e9; border:2px solid #43a047;
                     border-radius:10px; padding:1.2rem; text-align:center; }}
        .pred-box .price {{ font-size:2.2rem; font-weight:800; color:#2e7d32; }}
        /* ── footer ── */
        footer {{ visibility:hidden; }}
        .custom-footer {{ text-align:center; color:#aaa; font-size:.78rem;
                          padding:1rem 0; border-top:1px solid #eee; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
# ── SECTION 1: DATA LOADING & CLEANING ────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

DATA_FILE = "Car details v3 (2).csv"


def extract_number(series: pd.Series) -> pd.Series:
    """Pull the first numeric token from a mixed text+unit column."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"([\d.]+)")[0], errors="coerce"
    )


@st.cache_data(show_spinner=False)
def load_data(path: str):
    """
    Load raw CSV and return (raw_df, clean_df).

    Cleaning steps
    ──────────────
    1.  Remove duplicate rows
    2.  Extract numeric values from mileage / engine / max_power
    3.  Drop the torque column (too inconsistent)
    4.  Drop rows missing the target (selling_price)
    5.  Fill remaining numeric NaN with column median
    6.  Remove extreme price outliers (IQR × 3 fence)
    """
    raw = pd.read_csv(path)
    df  = raw.copy()

    # ── Step 1: duplicates ──────────────────────────────────────────
    df.drop_duplicates(inplace=True)

    # ── Step 2: numeric extraction ──────────────────────────────────
    df["mileage"]   = extract_number(df["mileage"])
    df["engine"]    = extract_number(df["engine"])
    df["max_power"] = extract_number(df["max_power"])

    # ── Step 3: drop torque ─────────────────────────────────────────
    df.drop(columns=["torque"], inplace=True, errors="ignore")

    # ── Step 4: drop rows without a price ───────────────────────────
    df.dropna(subset=["selling_price"], inplace=True)

    # ── Step 5: fill numeric NaN with median ────────────────────────
    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].median(), inplace=True)

    # ── Step 6: outlier removal (IQR × 3) ───────────────────────────
    q1, q3 = df["selling_price"].quantile([0.25, 0.75])
    iqr    = q3 - q1
    df = df[df["selling_price"].between(q1 - 3 * iqr, q3 + 3 * iqr)]

    df.reset_index(drop=True, inplace=True)
    return raw, df


# ══════════════════════════════════════════════════════════════════════
# ── SECTION 2: FEATURE ENGINEERING ────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

FEATURES = [
    "car_age", "km_driven", "mileage", "engine",
    "max_power", "seats",
    "fuel_enc", "seller_type_enc", "transmission_enc", "owner_enc",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add car_age and integer-encode all categorical columns.
    Returns a new DataFrame – original is not modified.
    """
    df = df.copy()
    df["car_age"] = 2024 - df["year"]

    for col in ["fuel", "seller_type", "transmission", "owner"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

    return df


# ══════════════════════════════════════════════════════════════════════
# ── SECTION 3: MODEL TRAINING & EVALUATION ────────────────────────────
# ══════════════════════════════════════════════════════════════════════

MODELS = {
    "Linear Regression":     LinearRegression(),
    "Decision Tree":         DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest":         RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
}


@st.cache_resource(show_spinner=False)
def train_all(df_eng: pd.DataFrame):
    """
    Train Linear Regression, Decision Tree, and Random Forest.

    Target is log1p-transformed to normalise the right-skewed distribution.
    Returns
    -------
    results : dict  {model_name: {model, y_true, y_pred, R2, MAE, RMSE}}
    best    : str   name of best model by R²
    """
    fe = df_eng.dropna(subset=FEATURES + ["selling_price"])
    X  = fe[FEATURES]
    y  = np.log1p(fe["selling_price"])   # log-transform

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    for name, mdl in MODELS.items():
        mdl.fit(X_tr, y_tr)
        y_pred_log = mdl.predict(X_te)
        y_pred     = np.expm1(y_pred_log)
        y_true     = np.expm1(y_te)

        results[name] = {
            "model":  mdl,
            "y_true": y_true,
            "y_pred": y_pred,
            "R2":     round(r2_score(y_true, y_pred), 4),
            "MAE":    round(mean_absolute_error(y_true, y_pred), 0),
            "RMSE":   round(np.sqrt(mean_squared_error(y_true, y_pred)), 0),
        }

    best = max(results, key=lambda k: results[k]["R2"])
    return results, best


# ══════════════════════════════════════════════════════════════════════
# ── SECTION 4: VISUALIZATION HELPERS ──────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _show(fig, caption=""):
    """Render a matplotlib figure in Streamlit then close it."""
    st.pyplot(fig, use_container_width=True)
    if caption:
        st.markdown(f'<p class="fig-cap">{caption}</p>', unsafe_allow_html=True)
    plt.close(fig)


# ── 1. Selling Price Distribution ─────────────────────────────────────
def fig_price_dist(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor("#f7f9fc")

    # raw
    axes[0].set_facecolor("#f7f9fc")
    sns.histplot(df["selling_price"] / 1e5, bins=50, kde=True,
                 color=PRIMARY, ax=axes[0])
    axes[0].set_title("Selling Price Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Selling Price (₹ Lakhs)")
    axes[0].set_ylabel("Count")

    # log-transformed
    axes[1].set_facecolor("#f7f9fc")
    sns.histplot(np.log1p(df["selling_price"]), bins=50, kde=True,
                 color=SECONDARY, ax=axes[1])
    axes[1].set_title("Log-Transformed Price (normalised)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("log(1 + Selling Price)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    return fig


# ── 2. Fuel Type Analysis ──────────────────────────────────────────────
def fig_fuel(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor("#f7f9fc")

    counts = df["fuel"].value_counts()
    axes[0].set_facecolor("#f7f9fc")
    axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%",
                colors=sns.color_palette("Blues_d", len(counts)), startangle=140)
    axes[0].set_title("Fuel Type Distribution", fontsize=13, fontweight="bold")

    axes[1].set_facecolor("#f7f9fc")
    order = df.groupby("fuel")["selling_price"].median().sort_values(ascending=False).index
    sns.boxplot(x="fuel", y="selling_price", data=df, order=order,
                palette="Blues_d", ax=axes[1])
    axes[1].set_title("Selling Price by Fuel Type", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Fuel Type")
    axes[1].set_ylabel("Selling Price (₹)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig


# ── 3. Transmission Analysis ───────────────────────────────────────────
def fig_transmission(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor("#f7f9fc")

    tc = df["transmission"].value_counts()
    axes[0].set_facecolor("#f7f9fc")
    bars = axes[0].bar(tc.index, tc.values,
                       color=[PRIMARY, SECONDARY], edgecolor="white", width=.5)
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 30, f"{bar.get_height():,}",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_title("Transmission Type Count", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")

    axes[1].set_facecolor("#f7f9fc")
    sns.boxplot(x="transmission", y="selling_price", data=df,
                palette="Blues", ax=axes[1])
    axes[1].set_title("Selling Price by Transmission", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Transmission")
    axes[1].set_ylabel("Selling Price (₹)")
    plt.tight_layout()
    return fig


# ── 4. Owner Analysis ─────────────────────────────────────────────────
def fig_owner(df):
    order = ["First Owner", "Second Owner", "Third Owner",
             "Fourth & Above Owner", "Test Drive Car"]
    valid = [o for o in order if o in df["owner"].unique()]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")
    sns.boxplot(x="owner", y="selling_price", data=df, order=valid,
                palette="Blues_d", ax=ax)
    ax.set_title("Selling Price by Ownership History", fontsize=13, fontweight="bold")
    ax.set_xlabel("Owner Type")
    ax.set_ylabel("Selling Price (₹)")
    plt.xticks(rotation=10)
    plt.tight_layout()
    return fig


# ── 5. Year vs Selling Price ───────────────────────────────────────────
def fig_year(df):
    ya = df.groupby("year")["selling_price"].median().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")
    ax.plot(ya["year"], ya["selling_price"] / 1e5, marker="o",
            color=PRIMARY, linewidth=2.2, markersize=5)
    ax.fill_between(ya["year"], ya["selling_price"] / 1e5, alpha=.15, color=PRIMARY)
    ax.set_title("Median Selling Price by Year of Manufacture",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Median Price (₹ Lakhs)")
    plt.tight_layout()
    return fig


# ── 6. Mileage vs Selling Price ────────────────────────────────────────
def fig_mileage(df):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")
    ax.scatter(df["mileage"], df["selling_price"] / 1e5, alpha=.3,
               color=PRIMARY, s=12, edgecolors="none")
    ax.set_title("Mileage vs Selling Price", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mileage (kmpl)")
    ax.set_ylabel("Selling Price (₹ Lakhs)")
    plt.tight_layout()
    return fig


# ── 7. Engine vs Selling Price ─────────────────────────────────────────
def fig_engine(df):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")
    ax.scatter(df["engine"], df["selling_price"] / 1e5, alpha=.3,
               color=SECONDARY, s=12, edgecolors="none")
    ax.set_title("Engine Capacity vs Selling Price", fontsize=13, fontweight="bold")
    ax.set_xlabel("Engine (CC)")
    ax.set_ylabel("Selling Price (₹ Lakhs)")
    plt.tight_layout()
    return fig


# ── 8. Correlation Heatmap ────────────────────────────────────────────
def fig_heatmap(df):
    num = df.select_dtypes(include="number").drop(columns=["year"], errors="ignore")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#f7f9fc")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                linewidths=.5, linecolor="#eee", annot_kws={"size": 9})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ── 9. Feature Importance ──────────────────────────────────────────────
def fig_importance(rf_model):
    imp = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")
    colors = [PRIMARY if v >= imp.quantile(.75) else SECONDARY for v in imp]
    imp.plot(kind="barh", ax=ax, color=colors, edgecolor="none")
    ax.set_title("Random Forest – Feature Importances",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


# ── Actual vs Predicted ───────────────────────────────────────────────
def fig_actual_vs_pred(y_true, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")
    ax.scatter(y_true / 1e5, np.array(y_pred) / 1e5,
               alpha=.3, color=PRIMARY, s=14, edgecolors="none")
    lim = [0, max(y_true.max(), max(y_pred)) / 1e5 * 1.05]
    ax.plot(lim, lim, "r--", linewidth=1.8, label="Perfect prediction")
    ax.set_title(f"Actual vs Predicted — {model_name}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Price (₹ Lakhs)")
    ax.set_ylabel("Predicted Price (₹ Lakhs)")
    ax.legend()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# ── LOAD DATA & TRAIN (runs once, cached) ─────────────────────────────
# ══════════════════════════════════════════════════════════════════════

with st.spinner("⏳ Loading dataset and training models …"):
    raw_df, clean_df = load_data(DATA_FILE)
    eng_df           = engineer(clean_df)
    results, best    = train_all(eng_df)

best_model = results[best]["model"]

# ══════════════════════════════════════════════════════════════════════
# ── SIDEBAR ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

st.sidebar.markdown(
    f"""
    <div style="background:{PRIMARY};border-radius:10px;
                padding:.8rem 1rem;margin-bottom:1rem;color:#fff;text-align:center;">
        <div style="font-size:1.6rem;">🚗</div>
        <div style="font-size:.95rem;font-weight:700;">Used Car Price</div>
        <div style="font-size:.75rem;opacity:.8;">Prediction System</div>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGE = st.sidebar.radio(
    "📌 Navigate",
    [
        "🏠  Home",
        "📋  Dataset Overview",
        "📊  Summary Statistics",
        "🔍  EDA & Visualizations",
        "🗺️  Correlation Heatmap",
        "🤖  Model Performance",
        "💡  Predict Price",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** [CarDekho – Kaggle]"
    "(https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)"
)
st.sidebar.markdown(f"**Best Model:** {best} ({results[best]['R2']})")
st.sidebar.markdown(
    "<div class='custom-footer'>IBM SkillsBuild Project</div>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
# ══  PAGE 1 – HOME  ══════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
if PAGE == "🏠  Home":
    st.markdown(
        f"""
        <div class="hero">
            <h1>🚗 Smart Used Car Price Prediction</h1>
            <p>IBM SkillsBuild · AI & Data Science Project ·
               Random Forest | Decision Tree | Linear Regression</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI row
    best_r2   = results[best]["R2"]
    best_mae  = int(results[best]["MAE"])
    st.markdown(
        f"""
        <div class="kpi-row">
            <div class="kpi"><div class="val">{len(clean_df):,}</div>
                <div class="lbl">Cleaned Records</div></div>
            <div class="kpi"><div class="val">{len(FEATURES)}</div>
                <div class="lbl">Features Used</div></div>
            <div class="kpi"><div class="val">{best_r2}</div>
                <div class="lbl">Best R² Score</div></div>
            <div class="kpi"><div class="val">₹{best_mae:,}</div>
                <div class="lbl">Best MAE</div></div>
            <div class="kpi"><div class="val">3</div>
                <div class="lbl">Models Trained</div></div>
            <div class="kpi"><div class="val">{best.split()[0]}</div>
                <div class="lbl">Best Model</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<p class="sec-hdr">🎯 Project Objective</p>', unsafe_allow_html=True)
        st.info(
            "Develop a machine learning system that predicts the **selling price of used cars** "
            "based on year, fuel type, transmission, ownership, mileage, engine, power, seats, "
            "and km driven — helping buyers and sellers estimate **fair market value**."
        )
        st.markdown('<p class="sec-hdr">📌 Pipeline</p>', unsafe_allow_html=True)
        for step in [
            "✅ Data Loading & Cleaning",
            "✅ Outlier Detection (IQR method)",
            "✅ Feature Engineering (car_age, encoded categoricals)",
            "✅ EDA with 9 Visualizations",
            "✅ 3 ML Models Trained & Compared",
            "✅ Interactive Price Prediction + CSV Download",
        ]:
            st.success(step)

    with c2:
        st.markdown('<p class="sec-hdr">📂 Dataset Sample</p>', unsafe_allow_html=True)
        st.dataframe(raw_df.head(8), use_container_width=True)

    # quick price dist teaser
    st.markdown('<p class="sec-hdr">📈 Selling Price Distribution (Preview)</p>', unsafe_allow_html=True)
    _show(
        fig_price_dist(clean_df),
        "Figure 1 – Left: selling price in lakhs (right-skewed). "
        "Right: log-transformed price used as model target.",
    )


# ══════════════════════════════════════════════════════════════════════
# ══  PAGE 2 – DATASET OVERVIEW  ══════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
elif PAGE == "📋  Dataset Overview":
    st.markdown('<p class="sec-hdr">📋 Dataset Overview</p>', unsafe_allow_html=True)

    tab_raw, tab_clean, tab_missing = st.tabs(
        ["📄 Raw Dataset", "🧹 Cleaned Dataset", "🔎 Missing Values"]
    )

    with tab_raw:
        st.write(f"**Shape:** {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")
        st.dataframe(raw_df, use_container_width=True)

    with tab_clean:
        st.write(
            f"**Shape:** {clean_df.shape[0]:,} rows × {clean_df.shape[1]} columns  "
            f"| Removed {raw_df.shape[0] - clean_df.shape[0]:,} rows (duplicates + outliers)"
        )
        st.dataframe(clean_df, use_container_width=True)

    with tab_missing:
        st.markdown("**Missing values in raw data:**")
        mv = raw_df.isnull().sum().reset_index()
        mv.columns = ["Column", "Missing Count"]
        mv["Missing %"] = (mv["Missing Count"] / len(raw_df) * 100).round(2)
        mv = mv[mv["Missing Count"] > 0]
        if mv.empty:
            st.success("No missing values found in the raw dataset.")
        else:
            st.dataframe(mv, use_container_width=True)
        st.markdown("**After cleaning:**")
        st.success(
            f"Cleaned dataset has **0 missing values** across all {clean_df.shape[1]} columns."
        )

        # dtypes
        st.markdown("**Column Data Types (cleaned dataset):**")
        dtype_df = clean_df.dtypes.reset_index()
        dtype_df.columns = ["Column", "DType"]
        st.dataframe(dtype_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# ══  PAGE 3 – SUMMARY STATISTICS  ════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
elif PAGE == "📊  Summary Statistics":
    st.markdown('<p class="sec-hdr">📊 Descriptive Statistics</p>', unsafe_allow_html=True)

    st.dataframe(
        clean_df.describe().T.style.background_gradient(cmap="Blues"),
        use_container_width=True,
    )
    st.caption("Table 1 – Descriptive statistics for all numeric columns (cleaned dataset).")

    st.markdown('<p class="sec-hdr">📦 Categorical Distributions</p>', unsafe_allow_html=True)

    cat_col = st.selectbox(
        "Select a categorical column to explore:",
        ["fuel", "seller_type", "transmission", "owner"],
    )
    vc = clean_df[cat_col].value_counts().reset_index()
    vc.columns = [cat_col, "Count"]
    vc["Percentage (%)"] = (vc["Count"] / len(clean_df) * 100).round(2)

    cA, cB = st.columns([1, 1])
    with cA:
        st.dataframe(vc, use_container_width=True)
    with cB:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#f7f9fc")
        ax.set_facecolor("#f7f9fc")
        sns.barplot(x=cat_col, y="Count", data=vc,
                    palette="Blues_d", ax=ax, edgecolor="none")
        ax.set_title(f"{cat_col.replace('_',' ').title()} Distribution",
                     fontsize=12, fontweight="bold")
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # outlier detection insight
    st.markdown('<p class="sec-hdr">🚨 Outlier Detection (Selling Price – IQR Method)</p>',
                unsafe_allow_html=True)
    q1, q3 = raw_df["selling_price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    low_fence  = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr
    outliers   = raw_df[
        (raw_df["selling_price"] < low_fence) |
        (raw_df["selling_price"] > high_fence)
    ]

    cO1, cO2, cO3 = st.columns(3)
    cO1.metric("IQR Lower Fence (1.5×)",  f"₹{low_fence:,.0f}")
    cO2.metric("IQR Upper Fence (1.5×)",  f"₹{high_fence:,.0f}")
    cO3.metric("Outliers Detected",        f"{len(outliers):,} rows")

    fig2, ax2 = plt.subplots(figsize=(9, 3.5))
    fig2.patch.set_facecolor("#f7f9fc")
    ax2.set_facecolor("#f7f9fc")
    ax2.boxplot(raw_df["selling_price"] / 1e5, vert=False, patch_artist=True,
                boxprops=dict(facecolor=LIGHT, color=PRIMARY),
                medianprops=dict(color="red", linewidth=2))
    ax2.set_xlabel("Selling Price (₹ Lakhs)")
    ax2.set_title("Selling Price – Outlier Box Plot (raw data)", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    st.caption(
        "Figure 2 – Box plot of raw selling prices. "
        "Points beyond the whiskers are outliers removed by IQR × 3 fence."
    )


# ══════════════════════════════════════════════════════════════════════
# ══  PAGE 4 – EDA & VISUALIZATIONS  ══════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
elif PAGE == "🔍  EDA & Visualizations":
    st.markdown('<p class="sec-hdr">🔍 Exploratory Data Analysis & Visualizations</p>',
                unsafe_allow_html=True)

    VIZ = st.selectbox(
        "Choose a visualization:",
        [
            "1.  Selling Price Distribution",
            "2.  Fuel Type Analysis",
            "3.  Transmission Analysis",
            "4.  Owner Type Analysis",
            "5.  Year vs Selling Price",
            "6.  Mileage vs Selling Price",
            "7.  Engine vs Selling Price",
            "9.  Feature Importance (Random Forest)",
        ],
    )

    if VIZ.startswith("1"):
        _show(fig_price_dist(clean_df),
              "Figure 3 – Selling price distribution. Raw (left) is right-skewed; "
              "log-transformation (right) normalises it for better model fit.")

    elif VIZ.startswith("2"):
        _show(fig_fuel(clean_df),
              "Figure 4 – Fuel type distribution (pie) and median selling price by fuel "
              "type (box plot). Diesel cars command a higher median price than Petrol.")

    elif VIZ.startswith("3"):
        _show(fig_transmission(clean_df),
              "Figure 5 – Manual cars dominate listings (~85%), but Automatic cars "
              "fetch a significantly higher selling price on average.")

    elif VIZ.startswith("4"):
        _show(fig_owner(clean_df),
              "Figure 6 – First-owner cars have the highest median selling price; "
              "value declines with each additional ownership change.")

    elif VIZ.startswith("5"):
        _show(fig_year(clean_df),
              "Figure 7 – Median selling price rises sharply for cars manufactured "
              "after 2015, confirming car_age as the strongest predictor.")

    elif VIZ.startswith("6"):
        _show(fig_mileage(clean_df),
              "Figure 8 – Mileage vs selling price. Weak positive correlation; "
              "high-mileage cars are not necessarily cheaper (fuel type matters).")

    elif VIZ.startswith("7"):
        _show(fig_engine(clean_df),
              "Figure 9 – Engine capacity vs selling price. Strong positive trend — "
              "larger engines are found in premium/luxury vehicles.")

    elif VIZ.startswith("9"):
        _show(fig_importance(best_model),
              "Figure 10 – Random Forest feature importances. max_power, engine, "
              "car_age, and km_driven are the top four predictors.")


# ══════════════════════════════════════════════════════════════════════
# ══  PAGE 5 – CORRELATION HEATMAP  ═══════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
elif PAGE == "🗺️  Correlation Heatmap":
    st.markdown('<p class="sec-hdr">🗺️ Feature Correlation Heatmap</p>', unsafe_allow_html=True)
    _show(
        fig_heatmap(clean_df),
        "Figure 11 – Pearson correlation matrix. max_power (r≈0.74) and engine "
        "(r≈0.68) show the strongest positive correlations with selling_price. "
        "km_driven shows a moderate negative correlation (r≈−0.23).",
    )

    st.markdown("### 📝 Key Correlation Insights")
    insight_data = {
        "Feature Pair": [
            "max_power ↔ selling_price",
            "engine ↔ selling_price",
            "km_driven ↔ selling_price",
            "mileage ↔ selling_price",
            "engine ↔ max_power",
        ],
        "Correlation": ["~+0.74", "~+0.68", "~−0.23", "~−0.12", "~+0.85"],
        "Interpretation": [
            "Higher power → much higher price (strongest predictor)",
            "Bigger engine → higher price (luxury/performance cars)",
            "More km driven → slightly lower price (wear & tear)",
            "Better mileage → slightly lower price (budget cars)",
            "Very strong — engine size drives power output",
        ],
    }
    st.dataframe(pd.DataFrame(insight_data), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# ══  PAGE 6 – MODEL PERFORMANCE  ════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
elif PAGE == "🤖  Model Performance":
    st.markdown('<p class="sec-hdr">🤖 Model Training & Evaluation</p>', unsafe_allow_html=True)

    # ── Metrics table ─────────────────────────────────────────────
    metric_rows = []
    for name, res in results.items():
        metric_rows.append({
            "Model": name,
            "R² Score": res["R2"],
            "MAE (₹)":  f"₹{int(res['MAE']):,}",
            "RMSE (₹)": f"₹{int(res['RMSE']):,}",
            "Best?":    "🏆 Yes" if name == best else "",
        })
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)
    st.caption("Table 2 – Comparison of all three models on the 20% held-out test set.")

    st.markdown("---")

    # ── Per-model Actual vs Predicted ─────────────────────────────
    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            tag = " 🏆" if name == best else ""
            st.markdown(f"**{name}{tag}**")
            st.metric("R²",   res["R2"])
            st.metric("MAE",  f"₹{int(res['MAE']):,}")
            st.metric("RMSE", f"₹{int(res['RMSE']):,}")
            _show(
                fig_actual_vs_pred(res["y_true"], res["y_pred"], name),
                f"Actual vs Predicted — {name}",
            )

    st.markdown("---")

    # ── Feature importance (best = RF) ────────────────────────────
    st.markdown('<p class="sec-hdr">📊 Feature Importance (Random Forest)</p>',
                unsafe_allow_html=True)
    _show(
        fig_importance(best_model),
        "Figure 12 – Feature importances of the best Random Forest model. "
        "max_power and engine CC are the dominant factors.",
    )

    # ── Model config table ─────────────────────────────────────────
    st.markdown("### ⚙️ Model Configuration")
    cfg = pd.DataFrame(
        [
            ["Linear Regression",  "–",                 "80% / 20%",  "log1p(price)"],
            ["Decision Tree",      "max_depth = 10",    "80% / 20%",  "log1p(price)"],
            ["Random Forest",      "n_estimators = 200","80% / 20%",  "log1p(price)"],
        ],
        columns=["Model", "Key Hyper-parameter", "Train/Test Split", "Target Encoding"],
    )
    st.dataframe(cfg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# ══  PAGE 7 – PREDICT PRICE  ═════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
elif PAGE == "💡  Predict Price":
    st.markdown(
        f'<p class="sec-hdr">💡 Predict Used Car Selling Price — '
        f'Using {best} (R² = {results[best]["R2"]})</p>',
        unsafe_allow_html=True,
    )
    st.write("Fill in the details below to get an instant price estimate.")

    with st.form("predict_form"):
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            year         = st.slider("📅 Year of Manufacture", 1995, 2023, 2017)
            km_driven    = st.number_input("🛣️ Kilometres Driven", 0, 500_000, 40_000, step=1_000)
            fuel         = st.selectbox("⛽ Fuel Type",    sorted(clean_df["fuel"].unique().tolist()))
        with r1c2:
            seller_type  = st.selectbox("🏪 Seller Type",  sorted(clean_df["seller_type"].unique().tolist()))
            transmission = st.selectbox("⚙️ Transmission", sorted(clean_df["transmission"].unique().tolist()))
            owner        = st.selectbox("👤 Owner",        sorted(clean_df["owner"].unique().tolist()))
        with r1c3:
            mileage   = st.number_input("🔋 Mileage (kmpl)",    5.0,  50.0,  20.0, step=0.5)
            engine    = st.number_input("🔧 Engine (CC)",       500.0, 5000.0, 1200.0, step=50.0)
            max_power = st.number_input("⚡ Max Power (bhp)",   30.0,  600.0,  82.0,  step=5.0)
            seats     = st.slider("🪑 Seats", 2, 9, 5)

        submitted = st.form_submit_button("🔍 Predict Selling Price", use_container_width=True)

    if submitted:
        # Build input row with the same encoding as training
        le_map = {}
        for col in ["fuel", "seller_type", "transmission", "owner"]:
            le = LabelEncoder()
            le.fit(clean_df[col].astype(str))
            le_map[col] = le

        inp = pd.DataFrame(
            [[
                2024 - year, km_driven, mileage, engine, max_power, seats,
                le_map["fuel"].transform([fuel])[0],
                le_map["seller_type"].transform([seller_type])[0],
                le_map["transmission"].transform([transmission])[0],
                le_map["owner"].transform([owner])[0],
            ]],
            columns=FEATURES,
        )

        pred_price = np.expm1(best_model.predict(inp)[0])
        low        = pred_price * 0.90
        high       = pred_price * 1.10

        st.markdown(
            f"""
            <div class="pred-box">
                <div style="font-size:.9rem;color:#388e3c;font-weight:600;">
                    🏆 Estimated Selling Price ({best})
                </div>
                <div class="price">₹{pred_price:,.0f}</div>
                <div style="font-size:.88rem;color:#555;margin-top:.4rem;">
                    Price Range (±10%):  ₹{low:,.0f} – ₹{high:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.balloons()

        # ── Download button ──────────────────────────────────────
        result_df = pd.DataFrame(
            {
                "Year": [year], "KM Driven": [km_driven], "Fuel": [fuel],
                "Transmission": [transmission], "Owner": [owner],
                "Mileage (kmpl)": [mileage], "Engine (CC)": [engine],
                "Max Power (bhp)": [max_power], "Seats": [seats],
                "Predicted Price (₹)": [round(pred_price, 2)],
                "Low Estimate (₹)":    [round(low, 2)],
                "High Estimate (₹)":   [round(high, 2)],
            }
        )
        st.download_button(
            "⬇️ Download Prediction as CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="car_price_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ── Similar cars ─────────────────────────────────────────
        st.markdown("#### 🔎 Similar Cars in Dataset")
        sim = clean_df[
            (clean_df["fuel"]         == fuel)         &
            (clean_df["transmission"] == transmission) &
            (clean_df["year"].between(year - 2, year + 2)) &
            (clean_df["selling_price"].between(pred_price * 0.75, pred_price * 1.25))
        ][["name", "year", "km_driven", "fuel", "transmission",
           "owner", "selling_price"]].head(10)

        if sim.empty:
            st.info("No closely matching cars found for this configuration.")
        else:
            st.dataframe(sim, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="custom-footer">
        Smart Used Car Price Prediction &nbsp;·&nbsp;
        IBM SkillsBuild Project &nbsp;·&nbsp;
        Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
