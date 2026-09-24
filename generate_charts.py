"""
generate_charts.py
──────────────────
Standalone script that loads the dataset, trains the 3 models, and saves
every dashboard chart as a high-resolution PNG for inclusion in the report.

Run:  python generate_charts.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ── output folder ──────────────────────────────────────────────────────
OUT = "report_charts"
os.makedirs(OUT, exist_ok=True)

PRIMARY   = "#1f4e79"
SECONDARY = "#2e75b6"
ACCENT    = "#4472c4"
LIGHT     = "#dce6f0"

FEATURES = [
    "car_age", "km_driven", "mileage", "engine",
    "max_power", "seats",
    "fuel_enc", "seller_type_enc", "transmission_enc", "owner_enc",
]

# ── helper ────────────────────────────────────────────────────────────
def save(fig, name, dpi=150):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path}")


def extract_number(s):
    return pd.to_numeric(s.astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")


# ── 1. load & clean ────────────────────────────────────────────────────
print("Loading data …")
raw = pd.read_csv("Car details v3 (2).csv")
df  = raw.copy()
df.drop_duplicates(inplace=True)
df["mileage"]   = extract_number(df["mileage"])
df["engine"]    = extract_number(df["engine"])
df["max_power"] = extract_number(df["max_power"])
df.drop(columns=["torque"], inplace=True, errors="ignore")
df.dropna(subset=["selling_price"], inplace=True)
for col in df.select_dtypes(include="number").columns:
    df[col].fillna(df[col].median(), inplace=True)
q1, q3 = df["selling_price"].quantile([0.25, 0.75])
iqr = q3 - q1
df = df[df["selling_price"].between(q1 - 3*iqr, q3 + 3*iqr)].reset_index(drop=True)
print(f"  cleaned: {len(df):,} rows")

# ── 2. feature engineering ────────────────────────────────────────────
eng = df.copy()
eng["car_age"] = 2024 - eng["year"]
for col in ["fuel", "seller_type", "transmission", "owner"]:
    le = LabelEncoder()
    eng[f"{col}_enc"] = le.fit_transform(eng[col].astype(str))

# ── 3. train models ───────────────────────────────────────────────────
print("Training models …")
X = eng[FEATURES]
y = np.log1p(eng["selling_price"])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree":     DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
}
results = {}
for name, mdl in models.items():
    mdl.fit(X_tr, y_tr)
    yp  = np.expm1(mdl.predict(X_te))
    yt  = np.expm1(y_te)
    results[name] = {
        "model": mdl,
        "y_true": yt, "y_pred": yp,
        "R2":  round(r2_score(yt, yp), 4),
        "MAE": round(mean_absolute_error(yt, yp), 0),
        "RMSE":round(np.sqrt(mean_squared_error(yt, yp)), 0),
    }
    print(f"  {name}: R²={results[name]['R2']}  MAE=₹{int(results[name]['MAE']):,}")

rf = models["Random Forest"]

# ══════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════

print("\nGenerating charts …")

# ── Chart 1: Price Distribution ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.patch.set_facecolor("#f7f9fc")
for ax in axes: ax.set_facecolor("#f7f9fc")
sns.histplot(df["selling_price"]/1e5, bins=50, kde=True, color=PRIMARY, ax=axes[0])
axes[0].set_title("Selling Price Distribution", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Selling Price (₹ Lakhs)"); axes[0].set_ylabel("Count")
sns.histplot(np.log1p(df["selling_price"]), bins=50, kde=True, color=SECONDARY, ax=axes[1])
axes[1].set_title("Log-Transformed Price", fontsize=13, fontweight="bold")
axes[1].set_xlabel("log(1 + Selling Price)"); axes[1].set_ylabel("Count")
plt.tight_layout(); save(fig, "fig01_price_distribution.png")

# ── Chart 2: Fuel Analysis ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.patch.set_facecolor("#f7f9fc")
for ax in axes: ax.set_facecolor("#f7f9fc")
counts = df["fuel"].value_counts()
axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%",
            colors=sns.color_palette("Blues_d", len(counts)), startangle=140)
axes[0].set_title("Fuel Type Distribution", fontsize=13, fontweight="bold")
order = df.groupby("fuel")["selling_price"].median().sort_values(ascending=False).index
sns.boxplot(x="fuel", y="selling_price", data=df, order=order, palette="Blues_d", ax=axes[1])
axes[1].set_title("Selling Price by Fuel Type", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Fuel Type"); axes[1].set_ylabel("Selling Price (₹)")
plt.xticks(rotation=15); plt.tight_layout(); save(fig, "fig02_fuel_analysis.png")

# ── Chart 3: Transmission ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.patch.set_facecolor("#f7f9fc")
for ax in axes: ax.set_facecolor("#f7f9fc")
tc = df["transmission"].value_counts()
bars = axes[0].bar(tc.index, tc.values, color=[PRIMARY, SECONDARY], edgecolor="white", width=.5)
for b in bars:
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+30,
                 f"{b.get_height():,}", ha="center", fontsize=11, fontweight="bold")
axes[0].set_title("Transmission Count", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Count")
sns.boxplot(x="transmission", y="selling_price", data=df, palette="Blues", ax=axes[1])
axes[1].set_title("Selling Price by Transmission", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Transmission"); axes[1].set_ylabel("Selling Price (₹)")
plt.tight_layout(); save(fig, "fig03_transmission.png")

# ── Chart 4: Owner ────────────────────────────────────────────────────
order = ["First Owner","Second Owner","Third Owner","Fourth & Above Owner","Test Drive Car"]
valid = [o for o in order if o in df["owner"].unique()]
fig, ax = plt.subplots(figsize=(11, 4.5))
fig.patch.set_facecolor("#f7f9fc"); ax.set_facecolor("#f7f9fc")
sns.boxplot(x="owner", y="selling_price", data=df, order=valid, palette="Blues_d", ax=ax)
ax.set_title("Selling Price by Ownership History", fontsize=13, fontweight="bold")
ax.set_xlabel("Owner Type"); ax.set_ylabel("Selling Price (₹)")
plt.xticks(rotation=10); plt.tight_layout(); save(fig, "fig04_owner.png")

# ── Chart 5: Year vs Price ────────────────────────────────────────────
ya = df.groupby("year")["selling_price"].median().reset_index()
fig, ax = plt.subplots(figsize=(11, 4.5))
fig.patch.set_facecolor("#f7f9fc"); ax.set_facecolor("#f7f9fc")
ax.plot(ya["year"], ya["selling_price"]/1e5, marker="o", color=PRIMARY, lw=2.2, ms=5)
ax.fill_between(ya["year"], ya["selling_price"]/1e5, alpha=.15, color=PRIMARY)
ax.set_title("Median Selling Price by Year", fontsize=13, fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Median Price (₹ Lakhs)")
plt.tight_layout(); save(fig, "fig05_year_vs_price.png")

# ── Chart 6: Mileage vs Price ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor("#f7f9fc"); ax.set_facecolor("#f7f9fc")
ax.scatter(df["mileage"], df["selling_price"]/1e5, alpha=.3, color=PRIMARY, s=12, edgecolors="none")
ax.set_title("Mileage vs Selling Price", fontsize=13, fontweight="bold")
ax.set_xlabel("Mileage (kmpl)"); ax.set_ylabel("Selling Price (₹ Lakhs)")
plt.tight_layout(); save(fig, "fig06_mileage_vs_price.png")

# ── Chart 7: Engine vs Price ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor("#f7f9fc"); ax.set_facecolor("#f7f9fc")
ax.scatter(df["engine"], df["selling_price"]/1e5, alpha=.3, color=SECONDARY, s=12, edgecolors="none")
ax.set_title("Engine Capacity vs Selling Price", fontsize=13, fontweight="bold")
ax.set_xlabel("Engine (CC)"); ax.set_ylabel("Selling Price (₹ Lakhs)")
plt.tight_layout(); save(fig, "fig07_engine_vs_price.png")

# ── Chart 8: Correlation Heatmap ──────────────────────────────────────
num = df.select_dtypes(include="number").drop(columns=["year"], errors="ignore")
corr = num.corr()
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#f7f9fc")
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax,
            linewidths=.5, linecolor="#eee", annot_kws={"size": 9})
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout(); save(fig, "fig08_correlation_heatmap.png")

# ── Chart 9: Feature Importance ───────────────────────────────────────
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#f7f9fc"); ax.set_facecolor("#f7f9fc")
colors = [PRIMARY if v >= imp.quantile(.75) else SECONDARY for v in imp]
imp.plot(kind="barh", ax=ax, color=colors, edgecolor="none")
ax.set_title("Random Forest – Feature Importances", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
plt.tight_layout(); save(fig, "fig09_feature_importance.png")

# ── Chart 10: Outlier Box Plot ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 3.5))
fig.patch.set_facecolor("#f7f9fc"); ax.set_facecolor("#f7f9fc")
ax.boxplot(raw["selling_price"]/1e5, vert=False, patch_artist=True,
           boxprops=dict(facecolor=LIGHT, color=PRIMARY),
           medianprops=dict(color="red", linewidth=2))
ax.set_xlabel("Selling Price (₹ Lakhs)")
ax.set_title("Selling Price – Outlier Box Plot (Raw Data)", fontweight="bold")
plt.tight_layout(); save(fig, "fig10_outlier_boxplot.png")

# ── Chart 11-13: Actual vs Predicted for each model ──────────────────
for i, (name, res) in enumerate(results.items(), 11):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor("#f7f9fc"); ax.set_facecolor("#f7f9fc")
    ax.scatter(res["y_true"]/1e5, np.array(res["y_pred"])/1e5,
               alpha=.3, color=PRIMARY, s=14, edgecolors="none")
    lim = [0, max(res["y_true"].max(), max(res["y_pred"]))/1e5*1.05]
    ax.plot(lim, lim, "r--", lw=1.8, label="Perfect prediction")
    ax.set_title(f"Actual vs Predicted — {name}\nR²={res['R2']}  MAE=₹{int(res['MAE']):,}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Price (₹ Lakhs)"); ax.set_ylabel("Predicted Price (₹ Lakhs)")
    ax.legend()
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    slug = name.lower().replace(" ","_")
    plt.tight_layout(); save(fig, f"fig{i:02d}_actual_vs_pred_{slug}.png")

# ── Chart 14: Model Comparison Bar Chart ─────────────────────────────
model_names = list(results.keys())
r2_vals  = [results[n]["R2"]         for n in model_names]
mae_vals = [results[n]["MAE"]/1e5    for n in model_names]  # in lakhs

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.patch.set_facecolor("#f7f9fc")
for ax in axes: ax.set_facecolor("#f7f9fc")

x = range(len(model_names))
bars1 = axes[0].bar(x, r2_vals, color=[PRIMARY, SECONDARY, ACCENT], edgecolor="none", width=.5)
axes[0].set_xticks(x); axes[0].set_xticklabels(model_names, rotation=10)
axes[0].set_ylim(0, 1.05)
axes[0].set_title("R² Score Comparison", fontsize=13, fontweight="bold")
axes[0].set_ylabel("R² Score")
for b, v in zip(bars1, r2_vals):
    axes[0].text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.4f}",
                 ha="center", fontsize=10, fontweight="bold")

bars2 = axes[1].bar(x, mae_vals, color=[PRIMARY, SECONDARY, ACCENT], edgecolor="none", width=.5)
axes[1].set_xticks(x); axes[1].set_xticklabels(model_names, rotation=10)
axes[1].set_title("MAE Comparison (₹ Lakhs)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("MAE (₹ Lakhs)")
for b, v in zip(bars2, mae_vals):
    axes[1].text(b.get_x()+b.get_width()/2, v+0.005, f"₹{v:.2f}L",
                 ha="center", fontsize=10, fontweight="bold")

plt.tight_layout(); save(fig, "fig14_model_comparison.png")

print(f"\n✅  All charts saved to: {os.path.abspath(OUT)}/")
