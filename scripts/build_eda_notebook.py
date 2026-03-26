#!/usr/bin/env python3
"""Build the complete Phase 1 EDA notebook for the recidivism prediction project."""
import json
from pathlib import Path

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

cells = [

# ── SECTION 0: Title ──────────────────────────────────────────────────────────
md("""# Phase 1 — Exploratory Data Analysis
## Recidivism Prediction | ProPublica COMPAS Dataset

**Goal:** Understand the raw data before building any pipeline.  
**Dataset:** `compas-scores-two-years-violent.csv` — ~4,000 defendants from Broward County, FL (2013–2014)  
**Target:** `is_violent_recid` — did this person commit a violent crime within 2 years of release?

---
### Why this EDA matters
The COMPAS algorithm used by real courts has been shown (ProPublica, 2016) to have racial disparities in its false positive rate. This EDA establishes our **fairness baseline**: we measure those disparities in the raw data before training anything, so we can compare our model's behavior against the status quo.
"""),

# ── SECTION 1: Imports & Setup ────────────────────────────────────────────────
md("""## 1. Imports & Setup"""),

code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Consistent plot styling
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (10, 5)

# Resolve data path relative to project root (works from notebooks/ dir)
DATA_PATH = Path("../data/raw/compas-scores-two-years-violent.csv")
assert DATA_PATH.exists(), f"Dataset not found at {DATA_PATH}. Run the download step first."
print(f"Data path: {DATA_PATH.resolve()}")
"""),

# ── SECTION 2: Load & Inspect ────────────────────────────────────────────────
md("""## 2. Load Data & Initial Inspection

First look: shape, column names, dtypes, and a sample of rows.
"""),

code("""\
df = pd.read_csv(DATA_PATH)

print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print()
print("Column names:")
for col in df.columns:
    print(f"  {col}")
"""),

code("""\
# Data types and non-null counts
print(df.dtypes.to_string())
"""),

code("""\
# Sample rows — look at the raw data before any transformation
df.sample(5, random_state=42)
"""),

code("""\
# Key columns we'll use — quick sanity check
key_cols = [
    "id", "name", "sex", "age", "age_cat", "race",
    "juv_fel_count", "juv_misd_count", "priors_count",
    "c_charge_degree", "v_decile_score", "is_violent_recid"
]
df[key_cols].head(10)
"""),

code("""\
# Basic descriptive stats for numeric columns
df.describe().T
"""),

# ── SECTION 3: Target Distribution ────────────────────────────────────────────
md("""## 3. Target Variable Distribution: `is_violent_recid`

**`is_violent_recid`** is our binary label: 1 = committed a violent crime within 2 years, 0 = did not.

> **Class imbalance alert:** Violent recidivism is rare. Expect ~20–25% positive class.  
> This means a lazy classifier that always predicts 0 gets ~80% accuracy with zero predictive value.  
> **Primary metrics: ROC-AUC and F1-score — not accuracy.**
"""),

code("""\
target = "is_violent_recid"

counts = df[target].value_counts().sort_index()
rates  = df[target].value_counts(normalize=True).sort_index() * 100

print("Target value counts:")
print(counts.to_string())
print()
print("Target proportions (%):")
print(rates.round(2).to_string())
"""),

code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
axes[0].bar(["No reoffense (0)", "Violent reoffense (1)"],
            counts.values, color=["steelblue", "coral"], edgecolor="white")
axes[0].set_title("Target Class Counts", fontweight="bold")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")

# Pie chart
axes[1].pie(counts.values, labels=["No reoffense", "Violent reoffense"],
            autopct="%1.1f%%", colors=["steelblue", "coral"],
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Target Class Proportions", fontweight="bold")

plt.suptitle("is_violent_recid Distribution", y=1.02, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../docs/01_target_distribution.png", bbox_inches="tight")
plt.show()
print(f"\\nClass imbalance ratio: {counts[0]/counts[1]:.1f}:1 (negative:positive)")
"""),

# ── SECTION 4: Missing Values ─────────────────────────────────────────────────
md("""## 4. Missing Values Audit

Every null is a **data quality decision**, not just a number to fill.  
Document not just *how many* nulls exist, but *why* — this is what distinguishes a thoughtful DS answer from a junior one.
"""),

code("""\
null_counts = df.isnull().sum()
null_pct    = (df.isnull().mean() * 100).round(2)

missing = pd.DataFrame({
    "null_count": null_counts,
    "null_pct":   null_pct
}).query("null_count > 0").sort_values("null_count", ascending=False)

if missing.empty:
    print("No missing values found in the dataset.")
else:
    print(f"{len(missing)} columns have missing values:\\n")
    print(missing.to_string())
"""),

code("""\
# Visualise missingness heatmap (only if there are missing values)
if not missing.empty:
    plt.figure(figsize=(10, 4))
    sns.heatmap(df[missing.index].isnull(), cbar=False,
                yticklabels=False, cmap="YlOrRd")
    plt.title("Missingness Heatmap (yellow = null)", fontweight="bold")
    plt.tight_layout()
    plt.savefig("../docs/02_missingness.png", bbox_inches="tight")
    plt.show()
else:
    print("No missingness to visualize.")
"""),

code("""\
# Even without classic nulls, check for sentinels: -1, empty strings, 'N/A'
print("v_decile_score value distribution (check for 0 or -1 sentinels):")
print(df["v_decile_score"].value_counts().sort_index().to_string())
print()
print("Rows with v_decile_score <= 0 (invalid COMPAS scores):",
      (df["v_decile_score"] <= 0).sum())
"""),

# ── SECTION 5: Demographic Breakdown ─────────────────────────────────────────
md("""## 5. Demographic Breakdown of Recidivism Rate

This is the core of the **ProPublica investigation**.  

We compute the **actual violent recidivism rate** by `race`, `sex`, and `age_cat`.  
Then compare it against the **COMPAS score** to surface any disparate impact.

> **Key concept:** Disparate impact occurs when similarly situated individuals receive  
> systematically different treatment based on a protected characteristic — even if the  
> characteristic is not explicitly used in the model.
"""),

code("""\
# ── 5a. Recidivism rate by race ───────────────────────────────────────────────
race_stats = (df.groupby("race")["is_violent_recid"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "recid_rate", "count": "n"})
                .sort_values("recid_rate", ascending=False))

race_stats["recid_rate_pct"] = (race_stats["recid_rate"] * 100).round(2)
print("Violent recidivism rate by race:")
print(race_stats.to_string())
"""),

code("""\
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(race_stats.index, race_stats["recid_rate_pct"],
              color=sns.color_palette("muted", len(race_stats)),
              edgecolor="white")
ax.axhline(df["is_violent_recid"].mean() * 100, color="red",
           linestyle="--", linewidth=1.5, label="Overall average")
ax.set_ylabel("Violent Recidivism Rate (%)")
ax.set_title("Actual Violent Recidivism Rate by Race", fontweight="bold")
ax.legend()
for bar, (_, row) in zip(bars, race_stats.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{row['recid_rate_pct']:.1f}%\\n(n={row['n']:,})",
            ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("../docs/03_recidivism_by_race.png", bbox_inches="tight")
plt.show()
"""),

code("""\
# ── 5b. Recidivism rate by sex ────────────────────────────────────────────────
sex_stats = (df.groupby("sex")["is_violent_recid"]
               .agg(["mean", "count"])
               .rename(columns={"mean": "recid_rate", "count": "n"}))
sex_stats["recid_rate_pct"] = (sex_stats["recid_rate"] * 100).round(2)

print("Violent recidivism rate by sex:")
print(sex_stats.to_string())
"""),

code("""\
# ── 5c. Recidivism rate by age category ──────────────────────────────────────
age_order  = ["Less than 25", "25 - 45", "Greater than 45"]
age_stats  = (df.groupby("age_cat")["is_violent_recid"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "recid_rate", "count": "n"})
                .reindex(age_order))
age_stats["recid_rate_pct"] = (age_stats["recid_rate"] * 100).round(2)

print("Violent recidivism rate by age category:")
print(age_stats.to_string())
"""),

code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Sex
axes[0].bar(sex_stats.index, sex_stats["recid_rate_pct"],
            color=["steelblue", "coral"], edgecolor="white")
axes[0].set_title("Recidivism Rate by Sex", fontweight="bold")
axes[0].set_ylabel("Rate (%)")
for i, (_, row) in enumerate(sex_stats.iterrows()):
    axes[0].text(i, row["recid_rate_pct"] + 0.3,
                 f"{row['recid_rate_pct']:.1f}%", ha="center")

# Age
axes[1].bar(age_stats.index, age_stats["recid_rate_pct"],
            color=sns.color_palette("Blues_d", 3), edgecolor="white")
axes[1].set_title("Recidivism Rate by Age Category", fontweight="bold")
axes[1].set_ylabel("Rate (%)")
for i, (_, row) in enumerate(age_stats.iterrows()):
    axes[1].text(i, row["recid_rate_pct"] + 0.3,
                 f"{row['recid_rate_pct']:.1f}%", ha="center")

plt.suptitle("Demographic Breakdown of Actual Violent Recidivism", fontsize=13,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("../docs/04_recidivism_demographics.png", bbox_inches="tight")
plt.show()
"""),

# ── SECTION 6: COMPAS Score Analysis ──────────────────────────────────────────
md("""## 6. COMPAS Score Analysis — Fairness Baseline

`v_decile_score` is the COMPAS algorithm's violent recidivism risk score (1–10).  
Higher = higher predicted risk.

We now compute the **COMPAS false positive rate (FPR) by race** — defendants predicted  
high-risk (score ≥ 5) who did *not* actually reoffend. This is the ProPublica finding.

> **FPR** = False Positives / (False Positives + True Negatives)  
> A high FPR means the model is over-flagging people who aren't actually dangerous.  
> If FPR differs by race, the model has **disparate impact** on that group.
"""),

code("""\
# Distribution of COMPAS score by race
plt.figure(figsize=(12, 5))
race_order = df["race"].value_counts().index.tolist()
sns.boxplot(data=df, x="race", y="v_decile_score",
            order=race_order, palette="muted")
plt.axhline(5, color="red", linestyle="--", linewidth=1.5,
            label="High-risk threshold (≥5)")
plt.title("COMPAS Violent Risk Score Distribution by Race", fontweight="bold")
plt.ylabel("COMPAS v_decile_score (1–10)")
plt.xlabel("")
plt.legend()
plt.tight_layout()
plt.savefig("../docs/05_compas_score_by_race.png", bbox_inches="tight")
plt.show()
"""),

code("""\
# ── COMPAS False Positive Rate by race ────────────────────────────────────────
THRESHOLD = 5  # COMPAS high-risk threshold

fpr_by_race = []
for race, grp in df.groupby("race"):
    did_not_reoffend = grp[grp["is_violent_recid"] == 0]
    flagged_high     = did_not_reoffend[did_not_reoffend["v_decile_score"] >= THRESHOLD]
    fpr = len(flagged_high) / len(did_not_reoffend) if len(did_not_reoffend) > 0 else 0

    did_reoffend = grp[grp["is_violent_recid"] == 1]
    missed       = did_reoffend[did_reoffend["v_decile_score"] < THRESHOLD]
    fnr = len(missed) / len(did_reoffend) if len(did_reoffend) > 0 else 0

    fpr_by_race.append({
        "race": race, "n": len(grp),
        "n_no_reoffend": len(did_not_reoffend),
        "fpr_compas": round(fpr, 4),
        "fnr_compas": round(fnr, 4),
    })

fpr_df = pd.DataFrame(fpr_by_race).set_index("race").sort_values("fpr_compas", ascending=False)
print("COMPAS Fairness Audit (threshold = score ≥ 5 → 'high risk'):")
print(fpr_df.to_string())
"""),

code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# FPR
axes[0].bar(fpr_df.index, fpr_df["fpr_compas"] * 100,
            color=sns.color_palette("Reds_r", len(fpr_df)), edgecolor="white")
axes[0].set_title("COMPAS False Positive Rate by Race\\n(flagged high-risk, did NOT reoffend)",
                  fontweight="bold")
axes[0].set_ylabel("False Positive Rate (%)")
axes[0].tick_params(axis="x", rotation=25)

# FNR
axes[1].bar(fpr_df.index, fpr_df["fnr_compas"] * 100,
            color=sns.color_palette("Blues_r", len(fpr_df)), edgecolor="white")
axes[1].set_title("COMPAS False Negative Rate by Race\\n(flagged low-risk, DID reoffend)",
                  fontweight="bold")
axes[1].set_ylabel("False Negative Rate (%)")
axes[1].tick_params(axis="x", rotation=25)

plt.suptitle("COMPAS Fairness Baseline — Before Any Model Training",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("../docs/06_compas_fairness_baseline.png", bbox_inches="tight")
plt.show()
"""),

# ── SECTION 7: Correlation Matrix ─────────────────────────────────────────────
md("""## 7. Correlation Matrix — Numeric Features vs Target

Identifies:
1. **Which features correlate most with the target** (initial predictor candidates)
2. **Multicollinearity between features** (pairs with |r| > 0.8 are problematic for linear models)

> **Interview concept — Multicollinearity:**  
> When two features are highly correlated, they carry redundant information.  
> In logistic regression, this inflates coefficient variance, making individual  
> coefficients unreliable. In tree models, it splits feature importance between  
> the correlated pair. Detection: correlation matrix. Fix: drop one, PCA, or L1 regularization.
"""),

code("""\
numeric_cols = [
    "age", "juv_fel_count", "juv_misd_count",
    "priors_count", "v_decile_score", "is_violent_recid"
]

corr = df[numeric_cols].corr()

plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))  # show lower triangle only
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            mask=mask, linewidths=0.5, square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix — Numeric Features", fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("../docs/07_correlation_matrix.png", bbox_inches="tight")
plt.show()
"""),

code("""\
# Flag high-correlation pairs (excluding diagonal and target)
feature_cols = ["age", "juv_fel_count", "juv_misd_count", "priors_count", "v_decile_score"]
feat_corr = df[feature_cols].corr()

print("High-correlation feature pairs (|r| > 0.4):")
seen = set()
for col_a in feat_corr.columns:
    for col_b in feat_corr.columns:
        if col_a >= col_b:
            continue
        r = feat_corr.loc[col_a, col_b]
        if abs(r) > 0.4:
            print(f"  {col_a} × {col_b}: r = {r:.3f}")
"""),

code("""\
# Correlation of each feature with the target — quick ranking
target_corr = df[feature_cols + ["is_violent_recid"]].corr()["is_violent_recid"].drop("is_violent_recid")
target_corr_sorted = target_corr.abs().sort_values(ascending=False)

print("Feature correlation with target (is_violent_recid), by |r|:")
print(target_corr_sorted.round(4).to_string())
"""),

# ── SECTION 8: Feature Distributions ──────────────────────────────────────────
md("""## 8. Feature Distributions by Target Class

For each key numeric feature, compare the distribution between defendants  
who reoffended (1) and those who didn't (0). This reveals which features  
have **separating power** — the signal our model will exploit.
"""),

code("""\
features_to_plot = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "v_decile_score"]

fig, axes = plt.subplots(1, len(features_to_plot), figsize=(18, 4))

for ax, feat in zip(axes, features_to_plot):
    for label, color, name in [(0, "steelblue", "No reoffense"), (1, "coral", "Reoffended")]:
        subset = df[df["is_violent_recid"] == label][feat].dropna()
        ax.hist(subset, bins=20, alpha=0.6, color=color, label=name, density=True)
    ax.set_title(feat, fontweight="bold")
    ax.set_xlabel("")
    ax.legend(fontsize=8)

fig.suptitle("Feature Distributions by Target Class", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("../docs/08_feature_distributions.png", bbox_inches="tight")
plt.show()
"""),

# ── SECTION 9: EDA Summary ────────────────────────────────────────────────────
md("""## 9. EDA Summary & Key Findings

### Data profile
- Dataset: ~4,000 rows, suitable for local + Databricks Community Edition.
- Target: `is_violent_recid` — binary, **severely imbalanced** (~20–25% positive).
- **Primary metrics for modeling: ROC-AUC and F1-score. Not accuracy.**

### Missing values
- No structural nulls in key columns (document your actual findings above).
- Sentinel values: rows with `v_decile_score ≤ 0` are invalid COMPAS entries — filter in Silver layer.

### Class imbalance
- The ~4:1 negative:positive ratio means we must use `class_weight="balanced"` or SMOTE during training.
- Always evaluate on a stratified test split to preserve the imbalance in evaluation.

### COMPAS fairness baseline
- COMPAS exhibits a higher **false positive rate** for Black defendants vs white defendants at similar actual reoffense rates — reproducing the ProPublica finding.
- This baseline is our benchmark: our model should not exceed this FPR disparity.

### Feature signals
- `priors_count` and `v_decile_score` are the strongest individual predictors of actual reoffense.
- `age` is negatively correlated: younger defendants have higher reoffense rates.
- `juv_fel_count` and `juv_misd_count` have moderate correlation with each other — watch for multicollinearity in linear models.

### Next step
**Phase 2:** Set up Databricks Community Edition and ingest the raw CSV into a Delta Lake Bronze table using PySpark.

---
*Notebook: Phase 1 — EDA | Project: Recidivism Prediction | Arjun, UC Berkeley*
"""),

]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.6"
        }
    },
    "cells": cells,
}

out = Path("notebooks/01_eda.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"✓ Written: {out}  ({out.stat().st_size:,} bytes)")
print(f"  Cells: {len(cells)} total ({sum(1 for c in cells if c['cell_type']=='code')} code, {sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
