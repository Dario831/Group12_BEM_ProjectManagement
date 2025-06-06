#!/usr/bin/env python3
"""
Performance Score (1-5) – Multiclass NN (PyTorch)
Logging, early stopping, plots, and summary.
"""

# ---------- CONFIG ----------
MODEL_NAME       = "FeedForwardClassifier"
TARGET           = "Performance_Score"
NUM_CLASSES      = 5                 # 1-5 → 0-4
HIDDEN_LAYERS    = [512, 256, 128, 64]
NUM_EPOCHS       = 150
BATCH_SIZE       = 256
LEARNING_RATE    = 1e-3
TEST_SIZE        = 0.15
VAL_SIZE         = 0.15
RANDOM_STATE     = 43
EARLY_STOP       = 20

DATASET_PATH     = "dataset.xlsx"
OUTPUT_ROOT      = f"Performance/{MODEL_NAME}"
# --------------------------------

import os, json, joblib, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- Model ----------
class FeedForward(nn.Module):
    def __init__(self, d_in, hidden, n_cls):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_cls))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ---------- Helpers ----------
def make_loader(X, y, bs, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

def eval_cls(model, loader, y_true, device):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(b.to(device)) for b,_ in loader]).cpu()
    preds = logits.argmax(1).numpy()
    return preds, {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro"))
    }

# ---------- Main ----------
def main():
    os.makedirs(f"{OUTPUT_ROOT}/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_ROOT}/test",  exist_ok=True)

    df = pd.read_excel(DATASET_PATH)
    df = df.drop(columns=[c for c in ['Employee_ID', 'Gender', 'Work_Hours_Per_Week', 'Projects_Handled',
       'Overtime_Hours', 'Sick_Days', 'Remote_Work_Frequency',
       'Training_Hours', 'Promotions', 'Employee_Satisfaction_Score',
       'Resigned'] if c in df.columns])

    # from 1-5 → 0-4
    y = df[TARGET].round().clip(1,5).astype(np.int64).values - 1
    X = df.drop(columns=[TARGET])
    np.random.seed(RANDOM_STATE)  
    X.iloc[:, 5] = X.iloc[:, 5].astype(float) * np.random.normal(loc=0.4, scale=0.01, size=len(X))
    cat = X.select_dtypes(include="object").columns.tolist()
    num = X.select_dtypes(exclude="object").columns.tolist()

    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat),
        ("num", StandardScaler(), num)
    ])

    Xp = preproc.fit_transform(X).astype(np.float32)
    d_in = Xp.shape[1]

    X_tf, X_te, y_tf, y_te = train_test_split(
        Xp, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    val_rel = VAL_SIZE / (1-TEST_SIZE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tf, y_tf, test_size=val_rel, random_state=RANDOM_STATE, stratify=y_tf)

    train_loader      = make_loader(X_tr,  y_tr,  BATCH_SIZE, shuffle=True)
    train_eval_loader = make_loader(X_tr,  y_tr,  BATCH_SIZE)
    val_loader        = make_loader(X_val, y_val, BATCH_SIZE)
    test_loader       = make_loader(X_te,  y_te, BATCH_SIZE)

    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedForward(d_in, HIDDEN_LAYERS, NUM_CLASSES).to(dev)
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit  = nn.CrossEntropyLoss()

    best_acc, no_imp, logs = 0.0, 0, []
    loss_hist, acc_hist = [], []

    # Training loop
    for ep in range(1, NUM_EPOCHS+1):
        model.train()
        tot_loss = 0
        for xb,yb in train_loader:
            xb,yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * xb.size(0)

        train_loss = tot_loss / len(X_tr)
        _, train_m = eval_cls(model, train_eval_loader, y_tr, dev)
        _, val_m   = eval_cls(model, val_loader,   y_val, dev)

        loss_hist.append(train_loss)
        acc_hist.append(train_m["accuracy"])

        log = (f"Epoch {ep:03d} | "
               f"Train Loss: {train_loss:.4f} | "
               f"Train Acc: {train_m['accuracy']:.3f} | "
               f"Val Acc: {val_m['accuracy']:.3f}")
        print(log)
        logs.append(log)

        if val_m["accuracy"] > best_acc:
            best_acc, no_imp = val_m["accuracy"], 0
            torch.save(model.state_dict(), Path(OUTPUT_ROOT)/"model.pt")
        else:
            no_imp += 1
            if no_imp >= EARLY_STOP:
                logs.append(f"Early stopping at epoch {ep}")
                print(f"Early stopping at epoch {ep}")
                break

    Path(f"{OUTPUT_ROOT}/train/train_log.txt").write_text("\n".join(logs))
    joblib.dump(preproc, Path(OUTPUT_ROOT)/"preprocessor.joblib")

    # Plot loss
    plt.figure(); plt.plot(loss_hist); plt.title("Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.savefig(Path(f"{OUTPUT_ROOT}/train/training_loss.png")); plt.close()
    # Plot accuracy
    plt.figure(); plt.plot(acc_hist); plt.title("Training Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.savefig(Path(f"{OUTPUT_ROOT}/train/training_accuracy.png")); plt.close()

    # Final metrics
    model.load_state_dict(torch.load(Path(OUTPUT_ROOT)/"model.pt"))
    _, train_m = eval_cls(model, train_eval_loader, y_tr, dev)
    _, val_m   = eval_cls(model, val_loader,   y_val, dev)
    preds_t, test_m = eval_cls(model, test_loader, y_te, dev)

    metrics = {"train": train_m, "val": val_m, "test": test_m}
    Path(f"{OUTPUT_ROOT}/test/test_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"True_Label": y_te+1, "Predicted_Label": preds_t+1})\
      .to_csv(Path(f"{OUTPUT_ROOT}/test/predictions.csv"), index=False)

    # Parity plot
    plt.figure(figsize=(5,5))
    plt.scatter(y_te+1, preds_t+1, s=10, alpha=0.6)
    plt.plot([1,5],[1,5],"--"); plt.xlim(1,5); plt.ylim(1,5)
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("Parity Plot")
    plt.savefig(Path(f"{OUTPUT_ROOT}/test/parity_plot.png")); plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_te, preds_t, labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues"); plt.colorbar(); plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.xticks(range(NUM_CLASSES), range(1,6)); plt.yticks(range(NUM_CLASSES), range(1,6))
    plt.savefig(Path(f"{OUTPUT_ROOT}/test/confusion_matrix.png")); plt.close()

    # Summary txt
    summary = (
        f"Best Val Acc : {best_acc:.4f}\n"
        f"Final Train Acc : {train_m['accuracy']:.4f}\n"
        f"Final Test Acc  : {test_m['accuracy']:.4f}\n"
        f"Final Test F1   : {test_m['f1_macro']:.4f}\n"
        f"Hidden Layers   : {HIDDEN_LAYERS}\n"
        f"Epochs Trained  : {len(loss_hist)}"
    )
    Path(f"{OUTPUT_ROOT}/test/summary.txt").write_text(summary)

    print("\nFinal metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
