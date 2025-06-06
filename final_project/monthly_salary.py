#!/usr/bin/env python3
"""
Monthly Salary Prediction – Feed-Forward NN (PyTorch)
EarlyStopping, logging, plots, extra metrics.
"""

# ---------- CONFIG ----------
MODEL_NAME      = "FeedForwardRegressor"
TARGET          = "Monthly_Salary"
HIDDEN_LAYERS   = [512, 256, 128, 64]
NUM_EPOCHS      = 300
BATCH_SIZE      = 256
LEARNING_RATE   = 1e-3
TEST_SIZE       = 0.15
VAL_SIZE        = 0.15
RANDOM_STATE    = 42
EARLY_STOP      = 20

DATASET_PATH    = "dataset.xlsx"
OUTPUT_ROOT     = f"MonthSalary/{MODEL_NAME}"
# --------------------------------

import os, json, joblib, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- Model ----------
class FeedForward(nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(1)

# ---------- Helpers ----------
def loader(X, y, bs, sh=False):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                      batch_size=bs, shuffle=sh)

def metrics_reg(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {
        "MAE":  float(mae),
        "RMSE": float(rmse),
        "R2":   float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred)/np.clip(y_true,1e-6,None))) * 100),
        "Pearson_r": float(pearsonr(y_true, y_pred).statistic)
    }

# ---------- Main ----------
def main():
    os.makedirs(f"{OUTPUT_ROOT}/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_ROOT}/test",  exist_ok=True)

    # --- Load data ---
    df = pd.read_excel(DATASET_PATH)
    df = df.drop(columns=[c for c in ['Employee_ID', 'Gender', 'Work_Hours_Per_Week', 'Projects_Handled',
       'Overtime_Hours', 'Sick_Days', 'Remote_Work_Frequency',
       'Training_Hours', 'Promotions', 'Employee_Satisfaction_Score',
       'Resigned'] if c in df.columns])

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(np.float32).values

    cat = X.select_dtypes(include="object").columns.tolist()
    num = X.select_dtypes(exclude="object").columns.tolist()
    num2 = np.random.choice([-2,-1, 0, 1,2], size=len(X), p=[0.0125, 0.0125, 0.95, 0.0125,0.0125])
    X.iloc[:, 5] = np.clip(X.iloc[:, 5] + num2, 1, 5)
    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat),
        ("num", StandardScaler(), num)
    ])

    Xp = preproc.fit_transform(X).astype(np.float32)
    d_in = Xp.shape[1]

    X_tf, X_te, y_tf, y_te = train_test_split(
        Xp, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    val_rel = VAL_SIZE / (1-TEST_SIZE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tf, y_tf, test_size=val_rel, random_state=RANDOM_STATE)

    train_loader = loader(X_tr, y_tr, BATCH_SIZE, True)
    train_eval   = loader(X_tr, y_tr, BATCH_SIZE)
    val_loader   = loader(X_val, y_val, BATCH_SIZE)
    test_loader  = loader(X_te, y_te, BATCH_SIZE)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedForward(d_in, HIDDEN_LAYERS).to(dev)
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit  = nn.MSELoss()

    # --- Training ---
    best_rmse, no_imp, logs = float("inf"), 0, []
    loss_hist, mae_hist = [], []

    for ep in range(1, NUM_EPOCHS+1):
        model.train(); tot_loss = 0
        for xb,yb in train_loader:
            xb,yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            tot_loss += loss.item() * xb.size(0)

        train_loss = tot_loss / len(X_tr)
        pred_tr = torch.cat([model(b.to(dev)).detach() for b,_ in train_eval]).cpu().numpy()
        train_mae = mean_absolute_error(y_tr, pred_tr)

        pred_val = torch.cat([model(b.to(dev)).detach() for b,_ in val_loader]).cpu().numpy()
        val_rmse = mean_squared_error(y_val, pred_val, squared=False)
        val_mae  = mean_absolute_error(y_val, pred_val)

        loss_hist.append(train_loss); mae_hist.append(train_mae)

        log = (f"Epoch {ep:03d} | Train Loss: {train_loss:.4f} | "
               f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")
        print(log); logs.append(log)

        if val_rmse < best_rmse:
            best_rmse, no_imp = val_rmse, 0
            torch.save(model.state_dict(), Path(OUTPUT_ROOT)/"model.pt")
        else:
            no_imp += 1
            if no_imp >= EARLY_STOP:
                logs.append(f"Early stopping at epoch {ep}")
                print(f"Early stopping at epoch {ep}")
                break

    # --- Save logs & plots ---
    Path(f"{OUTPUT_ROOT}/train/train_log.txt").write_text("\n".join(logs))
    plt.figure(); plt.plot(loss_hist); plt.title("Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.savefig(Path(f"{OUTPUT_ROOT}/train/training_loss.png")); plt.close()

    plt.figure(); plt.plot(mae_hist); plt.title("Training MAE"); plt.xlabel("Epoch"); plt.ylabel("MAE")
    plt.savefig(Path(f"{OUTPUT_ROOT}/train/training_mae.png")); plt.close()

    joblib.dump(preproc, Path(OUTPUT_ROOT)/"preprocessor.joblib")

    # --- Final evaluation ---
    model.load_state_dict(torch.load(Path(OUTPUT_ROOT)/"model.pt"))
    preds_tr = torch.cat([model(b.to(dev)).detach() for b,_ in train_eval]).cpu().numpy()
    preds_val= torch.cat([model(b.to(dev)).detach() for b,_ in val_loader]).cpu().numpy()
    preds_te = torch.cat([model(b.to(dev)).detach() for b,_ in test_loader]).cpu().numpy()

    train_m = metrics_reg(y_tr, preds_tr)
    val_m   = metrics_reg(y_val, preds_val)
    test_m  = metrics_reg(y_te, preds_te)

    metrics = {"train": train_m, "val": val_m, "test": test_m}
    Path(f"{OUTPUT_ROOT}/test/test_metrics.json").write_text(json.dumps(metrics, indent=2))

    pd.DataFrame({"True_Label": y_te, "Predicted_Label": preds_te})\
      .to_csv(Path(f"{OUTPUT_ROOT}/test/predictions.csv"), index=False)

    # Parity plot
    plt.figure(figsize=(5,5))
    plt.scatter(y_te, preds_te, s=10, alpha=0.6)
    lims=[y_te.min()*0.9, y_te.max()*1.1]; plt.plot(lims,lims,"--")
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("Parity Plot")
    plt.savefig(Path(f"{OUTPUT_ROOT}/test/parity_plot.png")); plt.close()

    # Residual histogram
    residuals = y_te - preds_te
    plt.figure(figsize=(6,4)); plt.hist(residuals, bins=30, alpha=0.7)
    plt.title("Residuals Histogram"); plt.xlabel("Residual"); plt.ylabel("Count")
    plt.savefig(Path(f"{OUTPUT_ROOT}/test/residuals_hist.png")); plt.close()

    # Summary txt
    summary = (
        f"Best Val RMSE: {best_rmse:.4f}\n"
        f"Train MAE : {train_m['MAE']:.4f}\n"
        f"Test  MAE : {test_m['MAE']:.4f}\n"
        f"Test  MAPE: {test_m['MAPE']:.2f}%\n"
        f"Test  Pearson r: {test_m['Pearson_r']:.4f}\n"
        f"Hidden Layers: {HIDDEN_LAYERS}\n"
        f"Epochs Trained: {len(loss_hist)}"
    )
    Path(f"{OUTPUT_ROOT}/test/summary.txt").write_text(summary)

    print("\nFinal metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
