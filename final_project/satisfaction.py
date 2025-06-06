#!/usr/bin/env python3
"""
Employee Satisfaction (3 classes) – CatBoost with:
- automatic class weights (“Balanced”)
- feature engineering (average performance by department)
- early stopping
- static hyperparameters
- scoring based on macro F1
- probability diagnostics and feature importances
- final plots (class distribution, confusion matrix)
"""
import torch
import pandas as pd
import numpy as np
import json, joblib
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
SEED        = 42
TARGET      = "Employee_Satisfaction_Score"
TEST_SIZE   = 0.20
VAL_SIZE    = 0.20
OUTDIR      = Path("Satisfaction/CatBoost_auto_weights")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------- Data -------------------
df = pd.read_excel("dataset.xlsx")
df = df.drop(columns=[c for c in ['Employee_ID', 'Gender',  'Projects_Handled',
       'Overtime_Hours', 'Sick_Days', 'Remote_Work_Frequency',
       'Training_Hours', 'Promotions', 'ID_progressive',
       'Resigned'] if c in df])

# binning into 3 classes
def bin_score(x):
    if x <= 2.40:   return "slightly"
    if x <= 4.08:  return "moderately"
    return "very"
df["SatisfactionBin"] = df[TARGET].apply(bin_score)

# feature engineering
if "Department" in df:
    df["DeptPerfMean"] = df.groupby("Department")["Performance_Score"].transform("mean")

# split X / y
y = df["SatisfactionBin"]
X = df.drop(columns=[TARGET, "SatisfactionBin"])

# stratified split
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, stratify=y, test_size=TEST_SIZE, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, stratify=y_tmp,
    test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=SEED
)

# class distribution diagnostics
print("Train distribution:", y_train.value_counts().to_dict())
print("Val   distribution:", y_val.value_counts().to_dict())
print("Test  distribution:", y_test.value_counts().to_dict())

# plot class distribution
def plot_dist(y_tr, y_va, y_te, labels, path):
    dfp = pd.concat([
        pd.DataFrame({"split":"Train","label":y_tr}),
        pd.DataFrame({"split":"Val",  "label":y_va}),
        pd.DataFrame({"split":"Test", "label":y_te})
    ])
    fig, ax = plt.subplots(figsize=(6,4))
    offsets = {"Train": -0.2, "Val": 0, "Test": 0.2}
    for split, grp in dfp.groupby("split"):
        counts = grp["label"].value_counts().reindex(labels, fill_value=0)
        ax.bar(np.arange(len(labels)) + offsets[split],
               counts, width=0.2, label=split)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Class distribution by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close()

LABELS = ["slightly", "moderately", "very"]
plot_dist(y_train, y_val, y_test, LABELS, OUTDIR / "class_distribution.png")

# prepare categorical features
cat_cols = X.select_dtypes(include="object").columns.tolist()
cat_idx  = [X.columns.get_loc(c) for c in cat_cols]

train_pool = Pool(data=X_train, label=y_train, cat_features=cat_idx)
val_pool   = Pool(data=X_val,   label=y_val,   cat_features=cat_idx)

# model with static hyperparameters
model = CatBoostClassifier(
    loss_function="MultiClass",
    auto_class_weights="Balanced",
    random_seed=SEED,
    verbose=False,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=5,
    iterations=1200
)

# training
model.fit(
    X_train, y_train,
    cat_features=cat_idx,
    eval_set=val_pool,
    early_stopping_rounds=200,
    verbose=False
)

# probability diagnostics
probs = model.predict_proba(X_test)
print("Avg predicted class probabilities:",
      dict(zip(LABELS, probs.mean(axis=0).round(3))))

# evaluation on test set
y_pred = model.predict(X_test)
if y_pred.ndim > 1:
    y_pred = y_pred.ravel()
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
cm     = confusion_matrix(y_test, y_pred, labels=LABELS)
torch.save(model, OUTDIR / "satisfaction.pt")

# save in joblib format
joblib.dump(model, OUTDIR / "preeprocessor_satisfaction.joblib")
params = {
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 5,
    "iterations": 1200
}
OUTDIR.joinpath("best_params.json").write_text(json.dumps(params, indent=2))
OUTDIR.joinpath("report.json").write_text(json.dumps(report, indent=2))
pd.DataFrame({"True": y_test.reset_index(drop=True), "Pred": pd.Series(y_pred).reset_index(drop=True)})\
  .to_csv(OUTDIR / "predictions.csv", index=False)

# plot confusion matrix
try:
    print("Confusion matrix shape:", cm.shape)
    cm = cm.T  # invert axes: True on X, Pred on Y
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(LABELS)),
        yticks=np.arange(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="True",
        ylabel="Predicted",
        title="Confusion Matrix – Test"
    )
    th = cm.max() / 2.
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > th else "black")
    fig.tight_layout()
    save_path = OUTDIR / "confusion_matrix.png"
    fig.savefig(save_path)
    print("Confusion matrix saved to:", save_path)
    plt.close()
except Exception as e:
    print("Error creating/saving confusion matrix:", str(e))

print("\nDone! All plots and outputs are in:", OUTDIR.resolve())
