"""
=============================================================
 IRIS FLOWER CLASSIFICATION — End-to-End ML Pipeline
=============================================================
 Pipeline Stages:
   1. Data Ingestion & Validation
   2. Exploratory Data Analysis (EDA)
   3. Data Preprocessing
   4. Feature Engineering
   5. Model Training & Hyperparameter Tuning
   6. Model Evaluation & Comparison
   7. Model Serialization (for deployment)
=============================================================
"""

# ─────────────────────────────────────────────────────────────
# STAGE 0 — Imports & Config
# ─────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import json
import os
from pathlib import Path

# Sklearn — Data
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Sklearn — Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Sklearn — Metrics
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

# Output directory
OUTPUT = Path("iris_outputs")
OUTPUT.mkdir(exist_ok=True)

SEED = 42
TARGET_NAMES = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

print("=" * 62)
print("  IRIS FLOWER CLASSIFICATION — ML PIPELINE")
print("=" * 62)


# ─────────────────────────────────────────────────────────────
# STAGE 1 — Data Ingestion & Validation
# ─────────────────────────────────────────────────────────────
print("\n[STAGE 1] Data Ingestion & Validation")
print("-" * 40)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=FEATURE_NAMES)
df["species"] = pd.Categorical.from_codes(iris.target, TARGET_NAMES)
df["species_id"] = iris.target

print(f"  ✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  ✓ Classes: {TARGET_NAMES}")
print(f"  ✓ Missing values: {df.isnull().sum().sum()}")
print(f"  ✓ Duplicates: {df.duplicated().sum()}")
print(f"\n  Class distribution:\n{df['species'].value_counts().to_string()}")
print(f"\n  Statistical summary:\n{df[FEATURE_NAMES].describe().round(2).to_string()}")

# Save raw data
df.to_csv(OUTPUT / "iris_raw.csv", index=False)
print(f"\n  ✓ Raw data saved → iris_outputs/iris_raw.csv")


# ─────────────────────────────────────────────────────────────
# STAGE 2 — Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────────────────────────
print("\n[STAGE 2] Exploratory Data Analysis")
print("-" * 40)

PALETTE = {"setosa": "#4FC3F7", "versicolor": "#81C784", "virginica": "#FF8A65"}

# --- Figure 1: Feature Distributions ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Feature Distributions by Species", fontsize=16, fontweight="bold", y=1.01)

for ax, feat in zip(axes.flatten(), FEATURE_NAMES):
    for species in TARGET_NAMES:
        subset = df[df["species"] == species][feat]
        ax.hist(subset, bins=15, alpha=0.65, label=species,
                color=PALETTE[species], edgecolor="white")
        ax.axvline(subset.mean(), color=PALETTE[species], lw=2, linestyle="--", alpha=0.9)
    ax.set_title(feat.replace("_", " ").title(), fontweight="bold")
    ax.set_xlabel("cm")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT / "eda_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved → eda_distributions.png")

# --- Figure 2: Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[FEATURE_NAMES].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            mask=mask, ax=ax, linewidths=0.5,
            cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(OUTPUT / "eda_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved → eda_correlation.png")

# --- Figure 3: Pairplot ---
pair_df = df[FEATURE_NAMES + ["species"]].copy()
g = sns.pairplot(pair_df, hue="species", palette=PALETTE,
                 diag_kind="kde", plot_kws={"alpha": 0.6, "s": 40})
g.fig.suptitle("Pairplot — Feature Relationships by Species",
               fontsize=14, fontweight="bold", y=1.02)
g.fig.savefig(OUTPUT / "eda_pairplot.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✓ Saved → eda_pairplot.png")

# --- Figure 4: Boxplots ---
fig, axes = plt.subplots(1, 4, figsize=(18, 6))
fig.suptitle("Boxplots — Feature Spread by Species", fontsize=15, fontweight="bold")
for ax, feat in zip(axes, FEATURE_NAMES):
    data_by_species = [df[df["species"] == s][feat].values for s in TARGET_NAMES]
    bp = ax.boxplot(data_by_species, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, species in zip(bp["boxes"], TARGET_NAMES):
        patch.set_facecolor(PALETTE[species])
        patch.set_alpha(0.8)
    ax.set_xticklabels([s.capitalize() for s in TARGET_NAMES], rotation=15)
    ax.set_title(feat.replace("_", " ").title(), fontweight="bold")
    ax.set_ylabel("cm")
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT / "eda_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved → eda_boxplots.png")


# ─────────────────────────────────────────────────────────────
# STAGE 3 — Preprocessing & Feature Engineering
# ─────────────────────────────────────────────────────────────
print("\n[STAGE 3] Preprocessing & Feature Engineering")
print("-" * 40)

# Feature Engineering — derived ratios
df["petal_area"]        = df["petal_length"] * df["petal_width"]
df["sepal_area"]        = df["sepal_length"] * df["sepal_width"]
df["petal_sepal_ratio"] = df["petal_length"] / df["sepal_length"]

ENGINEERED = FEATURE_NAMES + ["petal_area", "sepal_area", "petal_sepal_ratio"]
print(f"  ✓ Feature engineering: added petal_area, sepal_area, petal_sepal_ratio")

X = df[ENGINEERED].values
y = df["species_id"].values

# Train / Validation / Test split  (60 / 20 / 20)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp
)
print(f"  ✓ Split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)
print(f"  ✓ StandardScaler fitted on training set")

joblib.dump(scaler, OUTPUT / "scaler.pkl")
print(f"  ✓ Scaler saved → iris_outputs/scaler.pkl")


# ─────────────────────────────────────────────────────────────
# STAGE 4 — PCA Visualisation (2D & 3D)
# ─────────────────────────────────────────────────────────────
print("\n[STAGE 4] PCA Dimensionality Visualisation")
print("-" * 40)

pca2 = PCA(n_components=2, random_state=SEED)
X_pca2 = pca2.fit_transform(X_train_sc)

pca3 = PCA(n_components=3, random_state=SEED)
X_pca3 = pca3.fit_transform(X_train_sc)

fig = plt.figure(figsize=(16, 6))
colors = [list(PALETTE.values())[i] for i in y_train]

# 2D PCA
ax1 = fig.add_subplot(121)
for i, species in enumerate(TARGET_NAMES):
    mask = y_train == i
    ax1.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                c=PALETTE[species], label=species.capitalize(),
                s=70, alpha=0.8, edgecolors="white", linewidth=0.5)
ax1.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
ax1.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
ax1.set_title("PCA — 2D Projection", fontweight="bold")
ax1.legend()
ax1.spines[["top", "right"]].set_visible(False)

# 3D PCA
ax2 = fig.add_subplot(122, projection="3d")
for i, species in enumerate(TARGET_NAMES):
    mask = y_train == i
    ax2.scatter(X_pca3[mask, 0], X_pca3[mask, 1], X_pca3[mask, 2],
                c=PALETTE[species], label=species.capitalize(), s=50, alpha=0.7)
ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.set_zlabel("PC3")
ax2.set_title("PCA — 3D Projection", fontweight="bold")
ax2.legend()

plt.tight_layout()
plt.savefig(OUTPUT / "pca_projection.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved → pca_projection.png")
print(f"  ✓ Variance explained (2D): {sum(pca2.explained_variance_ratio_)*100:.1f}%")


# ─────────────────────────────────────────────────────────────
# STAGE 5 — Model Training & Comparison
# ─────────────────────────────────────────────────────────────
print("\n[STAGE 5] Model Training & Comparison")
print("-" * 40)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":       DecisionTreeClassifier(random_state=SEED),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=SEED),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=SEED),
    "SVM (RBF)":           SVC(kernel="rbf", probability=True, random_state=SEED),
    "Naive Bayes":         GaussianNB(),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = []

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring="accuracy")
    model.fit(X_train_sc, y_train)
    val_acc  = accuracy_score(y_val,  model.predict(X_val_sc))
    test_acc = accuracy_score(y_test, model.predict(X_test_sc))

    results.append({
        "Model":        name,
        "CV Mean":      cv_scores.mean(),
        "CV Std":       cv_scores.std(),
        "Val Accuracy": val_acc,
        "Test Accuracy":test_acc,
    })
    print(f"  {name:<22} CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}  "
          f"Val={val_acc:.4f}  Test={test_acc:.4f}")

results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
results_df.to_csv(OUTPUT / "model_comparison.csv", index=False)
print(f"\n  ✓ Model comparison saved → model_comparison.csv")


# ─────────────────────────────────────────────────────────────
# STAGE 6 — Hyperparameter Tuning (Best Model)
# ─────────────────────────────────────────────────────────────
print("\n[STAGE 6] Hyperparameter Tuning — Random Forest")
print("-" * 40)

param_grid = {
    "n_estimators":      [50, 100, 200],
    "max_depth":         [None, 3, 5, 10],
    "min_samples_split": [2, 5],
    "max_features":      ["sqrt", "log2"],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=SEED),
    param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0
)
grid_search.fit(X_train_sc, y_train)
best_model = grid_search.best_estimator_

print(f"  Best params: {grid_search.best_params_}")
print(f"  Best CV score: {grid_search.best_score_:.4f}")
print(f"  Test accuracy: {accuracy_score(y_test, best_model.predict(X_test_sc)):.4f}")


# ─────────────────────────────────────────────────────────────
# STAGE 7 — Evaluation & Reporting
# ─────────────────────────────────────────────────────────────
print("\n[STAGE 7] Final Evaluation & Reporting")
print("-" * 40)

y_pred      = best_model.predict(X_test_sc)
y_pred_prob = best_model.predict_proba(X_test_sc)

print("\n  Classification Report:")
report = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
print(report)

report_dict = classification_report(
    y_test, y_pred, target_names=TARGET_NAMES, output_dict=True
)
with open(OUTPUT / "classification_report.json", "w") as f:
    json.dump(report_dict, f, indent=2)

# --- Confusion Matrix ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=[s.capitalize() for s in TARGET_NAMES])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix — Random Forest", fontweight="bold")

# --- Feature Importance ---
feat_imp = pd.Series(best_model.feature_importances_, index=ENGINEERED).sort_values()
colors_bar = ["#4FC3F7" if f in FEATURE_NAMES else "#FF8A65" for f in feat_imp.index]
feat_imp.plot(kind="barh", ax=axes[1], color=colors_bar, edgecolor="white")
axes[1].set_title("Feature Importances", fontweight="bold")
axes[1].set_xlabel("Importance Score")
axes[1].spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT / "evaluation_cm_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved → evaluation_cm_features.png")

# --- Multi-class ROC Curves ---
y_test_bin  = label_binarize(y_test, classes=[0, 1, 2])
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = ["#4FC3F7", "#81C784", "#FF8A65"]
for i, (species, color) in enumerate(zip(TARGET_NAMES, colors_roc)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, color=color,
            label=f"{species.capitalize()} (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random Classifier")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — One-vs-Rest", fontweight="bold")
ax.legend(); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT / "evaluation_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved → evaluation_roc.png")


# ─────────────────────────────────────────────────────────────
# STAGE 8 — Model Comparison Chart
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
bar_w = 0.25
x = np.arange(len(results_df))

ax.bar(x - bar_w, results_df["CV Mean"],  bar_w, label="CV Mean",      color="#4FC3F7", alpha=0.85)
ax.bar(x,          results_df["Val Accuracy"], bar_w, label="Val Acc",  color="#81C784", alpha=0.85)
ax.bar(x + bar_w, results_df["Test Accuracy"], bar_w, label="Test Acc", color="#FF8A65", alpha=0.85)

for i, row in enumerate(results_df.itertuples()):
    ax.errorbar(i - bar_w, row._3, yerr=row._4, fmt="none",
                color="black", capsize=4, lw=1.5)

ax.set_xticks(x)
ax.set_xticklabels(results_df["Model"], rotation=25, ha="right", fontsize=10)
ax.set_ylabel("Accuracy"); ax.set_ylim(0.7, 1.05)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.legend(); ax.spines[["top", "right"]].set_visible(False)
ax.axhline(0.95, color="gray", lw=1, linestyle=":", alpha=0.6)
ax.text(len(results_df) - 0.5, 0.952, "95% threshold", fontsize=9, color="gray")
plt.tight_layout()
plt.savefig(OUTPUT / "model_comparison_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved → model_comparison_chart.png")


# ─────────────────────────────────────────────────────────────
# STAGE 9 — Serialization (Deployment Ready)
# ─────────────────────────────────────────────────────────────
print("\n[STAGE 9] Model Serialization")
print("-" * 40)

joblib.dump(best_model, OUTPUT / "iris_model.pkl")
metadata = {
    "model":         "RandomForestClassifier (GridSearchCV tuned)",
    "best_params":   grid_search.best_params_,
    "features":      ENGINEERED,
    "classes":       TARGET_NAMES,
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "cv_score":      float(grid_search.best_score_),
}
with open(OUTPUT / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  ✓ Model saved → iris_outputs/iris_model.pkl")
print(f"  ✓ Metadata saved → iris_outputs/model_metadata.json")


# ─────────────────────────────────────────────────────────────
# STAGE 10 — Inference Example
# ─────────────────────────────────────────────────────────────
print("\n[STAGE 10] Inference Demo")
print("-" * 40)

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    """Load model from disk and predict species."""
    model_  = joblib.load(OUTPUT / "iris_model.pkl")
    scaler_ = joblib.load(OUTPUT / "scaler.pkl")
    petal_area        = petal_length * petal_width
    sepal_area        = sepal_length * sepal_width
    petal_sepal_ratio = petal_length / sepal_length
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width,
                          petal_area, sepal_area, petal_sepal_ratio]])
    features_sc = scaler_.transform(features)
    pred    = model_.predict(features_sc)[0]
    proba   = model_.predict_proba(features_sc)[0]
    return TARGET_NAMES[pred], dict(zip(TARGET_NAMES, proba.round(4)))

samples = [
    (5.1, 3.5, 1.4, 0.2),   # setosa
    (6.0, 2.7, 5.1, 1.6),   # versicolor
    (6.3, 3.3, 6.0, 2.5),   # virginica
]
for s in samples:
    species, proba = predict_species(*s)
    print(f"  Input {s} → Predicted: {species.upper():<12} | Probabilities: {proba}")


print("\n" + "=" * 62)
print("  PIPELINE COMPLETE")
print("  All outputs saved to → iris_outputs/")
print("=" * 62)
