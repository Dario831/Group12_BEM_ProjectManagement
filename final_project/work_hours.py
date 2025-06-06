#!/usr/bin/env python3
"""
Work Hours per Week Prediction – Feed-Forward NN (PyTorch)
LR scheduler, EarlyStopping, logs & plots.
"""

# ---------- CONFIG ----------
MODEL_NAME       = "WorkHoursRegressor"
TARGET           = "Work_Hours_Per_Week"
HIDDEN_LAYERS    = [512, 256, 128, 64]
NUM_EPOCHS       = 400
BATCH_SIZE       = 256
INIT_LR          = 2e-3
TEST_SIZE        = 0.15
VAL_SIZE         = 0.15
RANDOM_STATE     = 42
EARLY_STOP       = 20
PLATEAU_PATIENCE = 8
PLATEAU_FACTOR   = 0.5

DATASET_PATH     = "dataset.xlsx"
OUTPUT_ROOT      = f"WorkHours/{MODEL_NAME}"
# --------------------------------

# Imports
import os, json, joblib, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ---------- Model ----------
class FeedForward(nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h),
                       nn.BatchNorm1d(h),
                       nn.ReLU(),
                       nn.Dropout(0.25)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(1)

# ---------- Helpers ----------
def mk_loader(X, y, bs, shuffle=False):
    return DataLoader(TensorDataset(torch.from_numpy(X),
                                    torch.from_numpy(y)),
                      batch_size=bs, shuffle=shuffle)

def reg_metrics(y_true, y_pred):
    return {
        "MAE":       float(mean_absolute_error(y_true, y_pred)),
        "RMSE":      float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2":        float(r2_score(y_true, y_pred)),
        "MAPE":      float(np.mean(np.abs((y_true - y_pred)/
                              np.clip(y_true, 1e-6, None))) * 100),
        "Pearson_r": float(pearsonr(y_true, y_pred).statistic)
    }

# ---------- Main ----------
def main():
    # Create folders
    (Path(OUTPUT_ROOT) / "train").mkdir(parents=True, exist_ok=True)
    (Path(OUTPUT_ROOT) / "test").mkdir(exist_ok=True)

    # Load data
    df = pd.read_excel(DATASET_PATH)
    df = df.drop(columns=[c for c in
                          ['Employee_ID', 'Gender', 'Projects_Handled',
       'Overtime_Hours', 'Sick_Days', 'Remote_Work_Frequency',
       'Training_Hours', 'Promotions', 'Employee_Satisfaction_Score',
       'Resigned']
                          if c in df.columns])

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(np.float32).values     # range ~ 20-60

    cat = X.select_dtypes(include="object").columns.tolist()
    num = X.select_dtypes(exclude="object").columns.tolist()
    x_pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat),
        ("num", StandardScaler(), num)
    ])

    Xp = x_pre.fit_transform(X).astype(np.float32)
    d_in = Xp.shape[1]

    # Train/Val/Test split 70/15/15
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        Xp, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=VAL_SIZE/(1-TEST_SIZE),
        random_state=RANDOM_STATE)

    train_loader = mk_loader(X_tr, y_tr, BATCH_SIZE, True)
    train_eval   = mk_loader(X_tr, y_tr, BATCH_SIZE)
    val_loader   = mk_loader(X_val, y_val, BATCH_SIZE)
    test_loader  = mk_loader(X_te, y_te, BATCH_SIZE)

    # Model definition
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedForward(d_in, HIDDEN_LAYERS).to(dev)
    opt   = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=PLATEAU_FACTOR,
                patience=PLATEAU_PATIENCE)
    crit  = nn.MSELoss()

    best_rmse, patience, logs = float("inf"), 0, []
    loss_hist, mae_hist = [], []

    # Training loop
    for ep in range(1, NUM_EPOCHS+1):
        model.train(); tot_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            tot_loss += loss.item() * xb.size(0)
        train_loss = tot_loss / len(X_tr)

        with torch.no_grad():
            pr_tr = torch.cat([model(b.to(dev))
                               for b,_ in train_eval]).cpu().numpy()
            pr_val= torch.cat([model(b.to(dev))
                               for b,_ in val_loader]).cpu().numpy()
        train_mae = mean_absolute_error(y_tr, pr_tr)
        val_rmse  = mean_squared_error(y_val, pr_val, squared=False)
        val_mae   = mean_absolute_error(y_val, pr_val)

        sched.step(val_rmse)
        loss_hist.append(train_loss); mae_hist.append(train_mae)

        logs.append(f"Ep {ep:03d} | loss {train_loss:.4f} | "
                    f"tMAE {train_mae:.3f} | vMAE {val_mae:.3f} | "
                    f"vRMSE {val_rmse:.3f}")
        print(logs[-1])

        if val_rmse < best_rmse:
            best_rmse, patience = val_rmse, 0
            torch.save(model.state_dict(), Path(OUTPUT_ROOT)/"model.pt")
        else:
            patience += 1
            if patience >= EARLY_STOP:
                logs.append(f"Early stopping at epoch {ep}")
                print(f"Early stopping at epoch {ep}")
                break

    # Save logs & plots
    Path(f"{OUTPUT_ROOT}/train/train_log.txt").write_text("\n".join(logs))
    plt.figure(); plt.plot(loss_hist); plt.title("Training Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.savefig(Path(f"{OUTPUT_ROOT}/train/training_loss.png")); plt.close()
    plt.figure(); plt.plot(mae_hist); plt.title("Training MAE")
    plt.xlabel("Epoch"); plt.ylabel("MAE")
    plt.savefig(Path(f"{OUTPUT_ROOT}/train/training_mae.png")); plt.close()

    joblib.dump(x_pre, Path(OUTPUT_ROOT)/"preprocessor.joblib")

    # Evaluate best model
    model.load_state_dict(torch.load(Path(OUTPUT_ROOT)/"model.pt"))
    with torch.no_grad():
        preds_tr = torch.cat([model(b.to(dev))
                     for b,_ in train_eval]).cpu().numpy()
        preds_val= torch.cat([model(b.to(dev))
                     for b,_ in val_loader]).cpu().numpy()
        preds_te = torch.cat([model(b.to(dev))
                     for b,_ in test_loader]).cpu().numpy()

    metrics = {
        "train": reg_metrics(y_tr, preds_tr),
        "val":   reg_metrics(y_val, preds_val),
        "test":  reg_metrics(y_te, preds_te)
    }
    Path(f"{OUTPUT_ROOT}/test/test_metrics.json").write_text(
        json.dumps(metrics, indent=2))

    pd.DataFrame({"True_Label": y_te, "Predicted_Label": preds_te})\
      .to_csv(Path(f"{OUTPUT_ROOT}/test/predictions.csv"), index=False)

    # Parity plot
    plt.figure(figsize=(5,5))
    plt.scatter(y_te, preds_te, s=10, alpha=0.6)
    lim = [min(y_te.min(), preds_te.min())-2,
           max(y_te.max(), preds_te.max())+2]
    plt.plot(lim, lim, "--"); plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("Parity Plot")
    plt.savefig(Path(f"{OUTPUT_ROOT}/test/parity_plot.png")); plt.close()

    # Residuals histogram
    residuals = y_te - preds_te
    plt.figure(figsize=(6,4)); plt.hist(residuals, bins=30, alpha=0.7)
    plt.title("Residuals Histogram"); plt.xlabel("Residual"); plt.ylabel("Count")
    plt.savefig(Path(f"{OUTPUT_ROOT}/test/residuals_hist.png")); plt.close()

    summary = (
        f"Best Val RMSE: {best_rmse:.3f}\n"
        f"Test MAE : {metrics['test']['MAE']:.3f}\n"
        f"Test MAPE: {metrics['test']['MAPE']:.2f}%\n"
        f"Test Pearson r: {metrics['test']['Pearson_r']:.3f}\n"
        f"Hidden Layers: {HIDDEN_LAYERS}\n"
        f"Epochs Trained: {len(loss_hist)}"
    )
    Path(f"{OUTPUT_ROOT}/test/summary.txt").write_text(summary)

    print("\nFinal metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
