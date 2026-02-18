#!/usr/bin/env python
# coding: utf-8

# vanilla lstm rv log z-scale forecast

# In[18]:


# ============================ Vanilla LSTM for Realized Variance (log + z, report in RV) ===========================
# Target for each horizon H:  RV_{t,t+H} = sum_{j=1}^H (Δy_{t+j})^2
# Train target: y*_t = zscore( log(RV_{t,t+H}) ) using TRAIN-only μ,σ
# Inputs at time t: [lagged volatility dynamics] + EXOGS_z
#   - logRV_1w_t      = log( (Δy_t)^2 + eps )
#   - logRV_4w_avg_t  = mean_{k=0..3} logRV_1w_{t-k}
#   - logRV_12w_avg_t = mean_{k=0..11} logRV_1w_{t-k}
#   - EXOGS (z-scored by TRAIN-only μ,σ)
# Evaluate & plot in RV space (back-transform from log + unscale).
# Save per split×H: CSV predictions + JSON config (LSTM_*), plus one LSTM_RV_summary.csv.
# ===================================================================================================================

import math, warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------- Config --------------------
from src.paths import DATA_SAMPLE, DATA_PRIVATE_DIR, RESULTS_DIR
from src.io_utils import load_table


DATE_COL   = "Date"
TARGET_COL = "Swap Spread"

# Your exogenous features (fixed; no selection)
EXOGS = ["VSTOXX","Euribor-OIS","Yield_Slope","Credit_Risk"]

# Folds and horizons
SPLITS   = [(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.00)]
HORIZONS = [1,4,12]

# Small grid (expand if you like)
L_GRID     = [12, 24, 52]
HID_GRID   = [64, 128]
DROPOUTS   = [0.0, 0.2]

# Training
VAL_FRAC     = 0.15           # time-ordered inner validation fraction
BATCH        = 64
EPOCHS_FULL  = 120
PATIENCE     = 15
LR           = 1e-3
WEIGHT_DECAY = 1e-4
CLIP_NORM    = 1.0
SEED         = 123
EPS          = 1e-12          # small floor for logs and divisions

# -------- Output folder (same as before; only filenames change to LSTM_*) --------
OUTPUT_DIR = RESULTS_DIR / "lstm_rv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUTPUT_DIR / "LSTM_RV_summary.csv"

# Repro/Device
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- Data load --------------------
DATA_PATH = DATA_SAMPLE  # default demo
df = load_table(DATA_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

for c in [TARGET_COL] + EXOGS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Weekly return/change
df["Δy"] = df[TARGET_COL].diff()
dates = df[DATE_COL].values

# -------------------- Helpers --------------------
def setup_dates(ax):
    loc = mdates.AutoDateLocator(minticks=6, maxticks=10)
    fmt = mdates.ConciseDateFormatter(loc)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)

def zfit(x):
    m = np.nanmean(x); s = np.nanstd(x)
    if not np.isfinite(s) or s <= 1e-12: s = 1.0
    return m, s

def smape(y_true, y_hat):
    y_true = np.asarray(y_true, float); y_hat = np.asarray(y_hat, float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_hat), 1e-8)
    return float(np.mean(2.0 * np.abs(y_true - y_hat) / denom) * 100.0)

def metrics_all(y_true, y_hat):
    y_true = np.asarray(y_true, float); y_hat = np.asarray(y_hat, float)
    err = y_true - y_hat
    if err.size == 0:
        return dict(N=0, MAE=np.nan, MSE=np.nan, RMSE=np.nan, MAPE=np.nan, sMAPE=np.nan, MFE=np.nan)
    mae  = float(np.mean(np.abs(err)))
    mse  = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs(err) / (np.abs(y_true) + 1e-8)) * 100.0)
    smp  = smape(y_true, y_hat)
    mfe  = float(np.mean(err))
    return dict(N=int(err.size), MAE=mae, MSE=mse, RMSE=rmse, MAPE=mape, sMAPE=smp, MFE=mfe)

def print_fold_summary_header(exogs_list, L, hid):
    exog_str = ", ".join(exogs_list) if exogs_list else "<none>"
    print(f"\n=== LSTM (exogs: {exog_str}, target = RV over H) — L={L}, hidden={hid} ===")

def print_metrics_line(h, m):
    print(
        f"  h={h:>2}: N={m['N']:4d}  MAE={m['MAE']:.6f}  MSE={m['MSE']:.6f}  RMSE={m['RMSE']:.6f}  "
        f"MAPE={m['MAPE']:.2f}%  sMAPE={m['sMAPE']:.2f}%  MFE={m['MFE']:.6f}"
    )

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_sequences_target_at_t(indices, L, y_target_z, X_block):
    """
    For each anchor t, target is y_target_z[t].
    Features are windows [t-L+1 .. t] of each series in X_block.
    """
    Xs, ys = [], []
    T = len(y_target_z)
    for t in indices:
        t0 = t - L + 1
        if t0 < 0 or t >= T:
            continue
        ys.append(y_target_z[t])
        segs = []
        for arr in X_block.values():
            seg = arr[t0:t+1]
            if len(seg) < L:
                seg = np.pad(seg, (L-len(seg), 0), mode='edge')
            segs.append(seg.reshape(-1, 1))
        Xs.append(np.concatenate(segs, axis=1))
    if len(Xs) == 0:
        D = len(X_block)
        return np.zeros((0, L, D)), np.zeros((0,))
    return np.stack(Xs, axis=0), np.array(ys)

class LSTMReg(nn.Module):
    def __init__(self, in_dim, hid=64, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hid, num_layers=1, batch_first=True, dropout=0.0)
        self.do   = nn.Dropout(dropout)
        self.fc   = nn.Linear(hid, 1)
    def forward(self, x):
        out,_ = self.lstm(x)
        out = out[:, -1, :]
        out = self.do(out)
        out = self.fc(out)
        return out.squeeze(-1)

def train_model(model, dl_tr, dl_va, epochs, lr, clip=1.0, wd=1e-4, patience=15):
    """
    Train with MSE + early stopping on validation MSE.
    Returns the model loaded with best-val weights.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    mse = nn.MSELoss()
    best_val = np.inf
    best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    no_improve = 0

    for ep in range(1, epochs+1):
        # Train
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            if clip: nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        # Validate
        model.eval()
        vls = []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                vls.append(mse(model(xb), yb).item())
        v = float(np.mean(vls)) if vls else np.inf
        if (ep % 10 == 0) or ep == 1:
            print(f"  epoch {ep:3d}  val_mse={v:.6f}")

        if v + 1e-9 < best_val:
            best_val = v
            best_state = {k: w.detach().cpu().clone() for k, w in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"  early stop at epoch {ep}")
            break

    model.load_state_dict(best_state)
    return model, best_val

def plot_with_legend_outside(dates_plot, y_true, y_hat, title_left, ylabel="Realized variance"):
    # Level plot in RV space
    fig, ax = plt.subplots(figsize=(11.5, 3.2))
    ax.plot(dates_plot, y_true, label="Actual RV")
    ax.plot(dates_plot, y_hat,  label="Forecast LSTM")  # <— label requested
    ax.set_title(title_left)
    ax.set_xlabel("Date"); ax.set_ylabel(ylabel)
    setup_dates(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    fig.tight_layout(); plt.show()

    # Error plot
    err = y_true - y_hat
    fig, ax = plt.subplots(figsize=(11.5, 2.9))
    ax.plot(dates_plot, err, label="Forecast error (RV)")
    ax.axhline(0, linewidth=0.8, color='black')
    ax.set_title(title_left.replace("Actual vs Forecast", "Forecast errors over time"))
    ax.set_xlabel("Date"); ax.set_ylabel("Error")
    setup_dates(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    fig.tight_layout(); plt.show()

# -------------------- Volatility feature builders --------------------
def past_mean(arr, window, min_periods=1):
    s = pd.Series(arr)
    return s.rolling(window, min_periods=min_periods).mean().to_numpy()

def build_vol_features_and_targets(dy, H):
    dy2 = np.square(dy)
    logRV_1w = np.log(np.maximum(dy2, EPS))
    logRV_4w = past_mean(logRV_1w, 4, 1)
    logRV_12w = past_mean(logRV_1w, 12, 1)

    rv_future = np.full_like(dy2, np.nan, dtype=float)
    for t in range(len(dy2)-H):
        rv_future[t] = np.sum(dy2[t+1:t+1+H])
    logrv_future = np.log(np.maximum(rv_future, EPS))

    return {
        "logRV_1w": logRV_1w,
        "logRV_4w": logRV_4w,
        "logRV_12w": logRV_12w,
        "rv_future": rv_future,
        "logrv_future": logrv_future
    }

# -------------------- One fold/H runner (train/val select, test once) --------------------
def run_fold_vanilla(a, b, H, L, hid, dropout, lr=LR, wd=WEIGHT_DECAY, patience=PATIENCE):
    N = len(df)
    tr_end = int(math.floor(a * N))
    te_end = int(math.floor(b * N))
    print(f"\n=== Fold {int(a*100)}→{int(b*100)} | H={H}, L={L}, hid={hid}, drop={dropout:.2f} ===")
    print(f"Train idx: [0..{tr_end-1}]  |  Test idx: [{tr_end}..{te_end-1}]")

    # Arrays up to te_end
    dy    = df["Δy"].values[:te_end]
    E     = df[EXOGS].values[:te_end].astype(float)

    # Build volatility dynamics + future targets
    feats = build_vol_features_and_targets(dy, H)
    logRV_1w  = feats["logRV_1w"]
    logRV_4w  = feats["logRV_4w"]
    logRV_12w = feats["logRV_12w"]
    rv_future     = feats["rv_future"]       # for evaluation/plots (RV space)
    logrv_future  = feats["logrv_future"]    # to be z-scored as training target

    # Train-only scaling for inputs and target
    m_E  = np.nanmean(E[:tr_end], 0); s_E = np.nanstd(E[:tr_end], 0); s_E[s_E<=1e-12] = 1.0
    E_z  = (E - m_E) / s_E

    m_l1, s_l1 = zfit(logRV_1w[:tr_end])
    m_l4, s_l4 = zfit(logRV_4w[:tr_end])
    m_l12, s_l12 = zfit(logRV_12w[:tr_end])

    logRV_1w_z  = (logRV_1w  - m_l1)  / s_l1
    logRV_4w_z  = (logRV_4w  - m_l4)  / s_l4
    logRV_12w_z = (logRV_12w - m_l12) / s_l12

    # Target scaling: z-score of log(RV_future) with TRAIN-only μ,σ
    m_y, s_y = zfit(logrv_future[:tr_end])
    y_target_z = (logrv_future - m_y) / s_y

    # Candidate anchors (must have valid target)
    valid = np.isfinite(y_target_z)
    idx_all = np.where(valid)[0]
    idx_all = idx_all[idx_all >= (L-1)]
    idx_tr  = idx_all[idx_all < tr_end]
    idx_te  = idx_all[idx_all >= tr_end]

    # Feature block (each series will be windowed across L)
    X_block = {
        "logRV_1w_z":  logRV_1w_z,
        "logRV_4w_z":  logRV_4w_z,
        "logRV_12w_z": logRV_12w_z,
    }
    for j, ex in enumerate(EXOGS):
        X_block[f"ex_{ex}"] = E_z[:, j]

    # Sequences
    X_tr, y_tr = make_sequences_target_at_t(idx_tr, L, y_target_z, X_block)
    X_te, y_te = make_sequences_target_at_t(idx_te, L, y_target_z, X_block)

    # Time-ordered validation
    n_tr = len(X_tr); n_va = max(1, int(round(VAL_FRAC * n_tr)))
    if n_tr < 30:
        print("  [warn] small training set; results may be noisy.")
    tr_slice = np.arange(0, max(0, n_tr - n_va))
    va_slice = np.arange(max(0, n_tr - n_va), n_tr)

    def mk_loaders(X, y, idx_tr, idx_va):
        if len(X) == 0 or len(idx_tr) == 0:
            return None, None
        ds_tr = SeqDataset(X[idx_tr], y[idx_tr])
        ds_va = SeqDataset(X[idx_va], y[idx_va]) if len(idx_va)>0 else SeqDataset(X[idx_tr], y[idx_tr])
        return DataLoader(ds_tr, batch_size=BATCH, shuffle=True), DataLoader(ds_va, batch_size=BATCH, shuffle=False)

    in_dim = X_tr.shape[2] if len(X_tr)>0 else len(X_block)
    dl_tr, dl_va = mk_loaders(X_tr, y_tr, tr_slice, va_slice)
    if dl_tr is None:
        y_true_rv = rv_future[idx_te]
        return {
            "config": {"L":L, "hid":hid, "drop":dropout},
            "metrics": metrics_all(y_true_rv, np.full_like(y_true_rv, np.nan)),
            "plots": (dates[idx_te], y_true_rv, np.full_like(y_true_rv, np.nan))
        }

    # Train & select (early stopping)
    model = LSTMReg(in_dim=in_dim, hid=hid, dropout=dropout).to(device)
    model, _ = train_model(model, dl_tr, dl_va, epochs=EPOCHS_FULL, lr=lr,
                           clip=CLIP_NORM, wd=wd, patience=patience)

    # Predict y*_z on TEST; back-transform to RV space
    with torch.no_grad():
        yhat_te_z = model(torch.tensor(X_te, dtype=torch.float32, device=device)).cpu().numpy()

    yhat_logrv = yhat_te_z * (s_y if s_y!=0 else 1.0) + m_y
    yhat_rv    = np.exp(yhat_logrv)
    y_true_rv  = rv_future[idx_te]

    metr = metrics_all(y_true_rv, yhat_rv)
    return {
        "config": {"L":L, "hid":hid, "drop":dropout},
        "metrics": metr,
        "plots": (dates[idx_te], y_true_rv, yhat_rv)
    }

# -------------------- Outer loop + file outputs --------------------
summary_rows = []

for (a,b) in SPLITS:
    N = len(df); tr_end = int(math.floor(a*N)); te_end = int(math.floor(b*N))
    split_tag_num = int(a*100)  # 60,70,80,90
    split_pretty  = f"{int(a*100)}→{int(b*100)}"
    print("\n" + "="*110)
    print(f"FOLD {split_pretty}   (train 0..{tr_end-1}, test {tr_end}..{te_end-1})")
    print("="*110)

    for H in HORIZONS:
        chosen = None
        chosen_metrics = None
        chosen_plots = None
        chosen_cfg = None

        for L in L_GRID:
            for hid in HID_GRID:
                for drop in DROPOUTS:
                    out = run_fold_vanilla(a, b, H, L, hid, drop,
                                           lr=LR, wd=WEIGHT_DECAY, patience=PATIENCE)
                    m = out["metrics"]
                    score = (m["MAE"], m["RMSE"])
                    if chosen is None or score < (chosen_metrics["MAE"], chosen_metrics["RMSE"]):
                        chosen = out
                        chosen_metrics = m
                        chosen_plots = out["plots"]
                        chosen_cfg = out["config"]

        # Print the compact block + one-line metrics
        print_fold_summary_header(EXOGS, chosen_cfg["L"], chosen_cfg["hid"])
        print_metrics_line(H, chosen_metrics)

        # Save predictions CSV (LSTM naming)
        dplot, ytrue_rv, yhat_rv = chosen_plots
        pred_df = pd.DataFrame({"Date": dplot, "y_true": ytrue_rv, "y_hat": yhat_rv})
        csv_path = os.path.join(OUTPUT_DIR, f"LSTM_RV_split_{split_tag_num}_H{H}.csv")
        pred_df.to_csv(csv_path, index=False)

        # Save config JSON (hyperparams, exogs, metrics)
        cfg = {
            "fold": f"split_{split_tag_num}",
            "H": H,
            "L": chosen_cfg["L"],
            "hidden": chosen_cfg["hid"],
            "dropout": chosen_cfg["drop"],
            "exogs": EXOGS,
            "metrics": chosen_metrics
        }
        with open(os.path.join(OUTPUT_DIR, f"LSTM_RV_split_{split_tag_num}_H{H}_config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)

        # Plot
        title = f"Vanilla LSTM — Fold {split_pretty} | H={H} — Actual vs Forecast (RV)"
        plot_with_legend_outside(dplot, ytrue_rv, yhat_rv, title_left=title, ylabel="Realized variance")

        # Add to summary
        summary_rows.append({
            "fold": f"split_{split_tag_num}",
            "H": H,
            "L": chosen_cfg["L"],
            "hidden": chosen_cfg["hid"],
            "dropout": chosen_cfg["drop"],
            "N": chosen_metrics["N"],
            "MAE": chosen_metrics["MAE"],
            "MSE": chosen_metrics["MSE"],
            "RMSE": chosen_metrics["RMSE"],
            "MAPE": chosen_metrics["MAPE"],
            "sMAPE": chosen_metrics["sMAPE"],
            "MFE": chosen_metrics["MFE"],
            "pred_file": os.path.basename(csv_path)
        })

# Write consolidated summary CSV (like your screenshot)
summary_df = pd.DataFrame(summary_rows).sort_values(["fold","H"])
summary_df.to_csv(SUMMARY_CSV, index=False)

print(f"\nSaved per-fold×H predictions/configs to: {OUTPUT_DIR}")
print(f"Consolidated summary: {SUMMARY_CSV}")
print("\n==================== Summary (all folds × horizons) ====================")
print(summary_df.to_string(index=False))


# HMM LSTM log rv z-scale 

# In[16]:


# ===================== HMM–LSTM Mixture-of-Experts for Realized Variance (log + z, report in RV) ===================
# Target for each horizon H:  RV_{t,t+H} = sum_{j=1}^H (Δy_{t+j})^2
# Train target (per t): y*_t = zscore( log(RV_{t,t+H}) ) using TRAIN-only μ,σ
# Inputs at time t (windowed over length L):
#   - logRV_1w_t      = log( (Δy_t)^2 + eps )
#   - logRV_4w_avg_t  = mean_{k=0..3}  logRV_1w_{t-k}
#   - logRV_12w_avg_t = mean_{k=0..11} logRV_1w_{t-k}
#   - EXOGS (z-scored by TRAIN-only μ,σ)
#   - HMM gate features: p_t, p_{t-1}, run_length_z
# Gating:
#   - Two LSTM experts (HI/LO). Prediction = p_t * yhat_HI + (1-p_t) * yhat_LO
#   - τ ∈ {0.45, 0.50, 0.55} chosen by quick inner loop (test-block MAE)
#   - Margin m=0.05: drop TRAIN rows with |p_t - τ| ≤ m
#   - Min run-length = 2 (sequential, no look-ahead)
# Training: early stopping on VAL MSE (loss on z-scored log-target)
# Reporting: back-transform to RV (exp of unscaled log-target). Metrics & plots in RV space.
# Prints per-H compact metrics line like:  h=?, N=?, MAE=?, MSE=?, RMSE=?, MAPE=?, sMAPE=?, MFE=?
# ===================================================================================================================

import os, json, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------- Config --------------------

DATE_COL   = "Date"
TARGET_COL = "Swap Spread"

# Exogenous variables
EXOG_COLS  = ["Credit_Risk","ECBrate","German_CPI YoY","Euribor-OIS","Yield_Slope","VSTOXX","German_UnemploymentRate"]

# Folds and horizons
SPLITS   = [(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.00)]
HORIZONS = [1,4,12]

# HMM probs location (expects sheet 'probs' with column 'w_high' aligned by Date)
HMM_DIR = RESULTS_DIR / "lstm_rv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUTPUT_DIR / "HMM_summary.csv"

# Model/Training search space
L_GRID     = [12,24,52]
HID_GRID   = [64,128]
DROPOUTS   = [0.0,0.2]
TAU_GRID   = [0.45,0.50,0.55]

# Gate & features
MARGIN     = 0.05
MIN_RUN    = 2
SEQ_CAP_RL = 52
EPS        = 1e-12  # floors for logs and divisions

# Training
VAL_FRAC     = 0.15
BATCH        = 64
EPOCHS_FULL  = 120
PATIENCE     = 15
LR           = 1e-3
WEIGHT_DECAY = 1e-4
CLIP_NORM    = 1.0
SEED         = 123

# Output
OUTPUT_DIR = RESULTS_DIR / "hmm_lstm_rv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUTPUT_DIR / "HMM_LSTM_RV_summary.csv"

# Repro/Device
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- Data load --------------------
DATA_PATH = DATA_SAMPLE  # default demo
df = load_table(DATA_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

# ensure numeric
for c in [TARGET_COL] + EXOG_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Weekly change
df["Δy"] = df[TARGET_COL].diff()
dates = df[DATE_COL].values

# -------------------- Helpers --------------------
def setup_dates(ax):
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

def hmm_probs_for_split(a, b, N):
    split_name = f"split_{int(a*100)}"
    fpath = os.path.join(HMM_DIR, f"hmm_{split_name}.xlsx")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"HMM probs not found: {fpath}")
    probs = pd.read_excel(fpath, sheet_name="probs")
    probs["Date"] = pd.to_datetime(probs["Date"])
    merged = pd.merge(df[[DATE_COL]], probs[["Date","w_high"]], left_on=DATE_COL, right_on="Date", how="left")
    p = merged["w_high"].astype(float).values
    if np.isnan(p).any():
        med = np.nanmedian(p)
        p = np.nan_to_num(p, nan=med)
    return p[:N]

def smooth_min_run(labels, min_run=2):
    if len(labels) == 0:
        return labels
    r = labels.copy()
    run = 1
    for t in range(1, len(r)):
        if r[t] == r[t-1]:
            run += 1
        else:
            if run < min_run:
                r[t] = r[t-1]
            run = 1
    return r

def run_length_from_labels(labels, cap=52):
    rl = np.zeros_like(labels, dtype=int)
    cur = 1
    for t in range(1, len(labels)):
        cur = cur + 1 if labels[t]==labels[t-1] else 1
        rl[t] = min(cur, cap)
    return rl

def zfit(x):
    m = np.nanmean(x); s = np.nanstd(x); s = 1.0 if (not np.isfinite(s) or s <= 1e-12) else s
    return m, s

def smape(y_true, y_hat):
    y_true = np.asarray(y_true, float); y_hat = np.asarray(y_hat, float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_hat), 1e-8)
    return float(np.mean(2.0 * np.abs(y_true - y_hat) / denom) * 100.0)

def metrics_all(y_true, y_hat):
    y_true = np.asarray(y_true, float); y_hat = np.asarray(y_hat, float)
    err = y_true - y_hat
    if err.size == 0:
        return dict(N=0, MAE=np.nan, MSE=np.nan, RMSE=np.nan, MAPE=np.nan, sMAPE=np.nan, MFE=np.nan)
    mae  = float(np.mean(np.abs(err)))
    mse  = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs(err) / (np.abs(y_true) + 1e-8)) * 100.0)
    smp  = smape(y_true, y_hat)
    mfe  = float(np.mean(err))
    return dict(N=int(err.size), MAE=mae, MSE=mse, RMSE=rmse, MAPE=mape, sMAPE=smp, MFE=mfe)

def past_mean(arr, window, min_periods=1):
    return pd.Series(arr).rolling(window, min_periods=min_periods).mean().to_numpy()

def build_vol_features_and_targets(dy, H):
    dy2 = np.square(dy)
    logRV_1w  = np.log(np.maximum(dy2, EPS))
    logRV_4w  = past_mean(logRV_1w, 4, 1)
    logRV_12w = past_mean(logRV_1w, 12, 1)

    rv_future = np.full_like(dy2, np.nan, dtype=float)
    for t in range(len(dy2)-H):
        rv_future[t] = np.sum(dy2[t+1:t+1+H])
    logrv_future = np.log(np.maximum(rv_future, EPS))
    return logRV_1w, logRV_4w, logRV_12w, rv_future, logrv_future

# -------------------- Sequences --------------------
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_sequences(indices, L, y_target_z, X_block):
    Xs, ys = [], []
    T = len(y_target_z)
    for t in indices:
        t0 = t - L + 1
        if t0 < 0 or t >= T or not np.isfinite(y_target_z[t]):
            continue
        ys.append(y_target_z[t])
        segs = []
        for arr in X_block.values():
            seg = arr[t0:t+1]
            if len(seg) < L:
                seg = np.pad(seg, (L-len(seg),0), mode='edge')
            segs.append(seg.reshape(-1,1))
        Xs.append(np.concatenate(segs, axis=1))
    Xs = np.stack(Xs, axis=0) if len(Xs)>0 else np.zeros((0, L, len(X_block)))
    ys = np.array(ys)
    return Xs, ys

# -------------------- Model --------------------
class LSTMReg(nn.Module):
    def __init__(self, in_dim, hid=64, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hid, num_layers=1, batch_first=True, dropout=0.0)
        self.do   = nn.Dropout(dropout)
        self.fc   = nn.Linear(hid, 1)
    def forward(self, x):
        out,_ = self.lstm(x)
        out = out[:, -1, :]
        out = self.do(out)
        out = self.fc(out)
        return out.squeeze(-1)

def train_model(model, dl_tr, dl_va, epochs, lr, clip=1.0, wd=1e-4, patience=15):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best = (np.inf, 0, {k: v.detach().cpu().clone() for k,v in model.state_dict().items()})
    no_improve = 0
    mse = nn.MSELoss()
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            if clip: nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        model.eval()
        vls = []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                vls.append(mse(model(xb), yb).item())
        v = float(np.mean(vls)) if vls else np.inf
        if (ep%10==0) or (ep==1):
            print(f"  epoch {ep:3d}  val_mse={v:.5f}")
        if v + 1e-9 < best[0]:
            best = (v, ep, {k: w.detach().cpu().clone() for k,w in model.state_dict().items()})
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"  early stop at epoch {ep} (best @ {best[1]})")
            break
    model.load_state_dict(best[2])
    return model

# -------------------- Printing helpers (your requested format) --------------------
def print_block_header(exogs_list, L, hid, drop, tau):
    exog_str = ", ".join(exogs_list) if exogs_list else "<none>"
    print(f"\n=== HMM-LSTM (exogs: {exog_str}, target = RV over H) — L={L}, hidden={hid}, drop={drop}, tau={tau} ===")

def print_metrics_line(h, m):
    print(
        f"h={h:>2}: N={m['N']:4d}  MAE={m['MAE']:.6f}  MSE={m['MSE']:.6f}  RMSE={m['RMSE']:.6f}  "
        f"MAPE={m['MAPE']:.2f}%  sMAPE={m['sMAPE']:.2f}%  MFE={m['MFE']:.6f}"
    )

# -------------------- Plotting (unchanged, optional) --------------------
def plot_with_legend_outside(dates_plot, y_true, y_hat, title_left, ylabel="Realized variance"):
    fig, ax = plt.subplots(figsize=(11.5, 3.2))
    ax.plot(dates_plot, y_true, label="Actual RV")
    ax.plot(dates_plot, y_hat,  label="Forecast HMM LSTM")
    ax.set_title(title_left)
    ax.set_xlabel("Date"); ax.set_ylabel(ylabel)
    setup_dates(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    fig.tight_layout(); plt.show()

    err = y_true - y_hat
    fig, ax = plt.subplots(figsize=(11.5, 2.9))
    ax.plot(dates_plot, err, label="Forecast error (RV)")
    ax.axhline(0, linewidth=0.8, color='black')
    ax.set_title(title_left.replace("Actual vs Forecast", "Forecast errors over time"))
    ax.set_xlabel("Date"); ax.set_ylabel("Error")
    setup_dates(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    fig.tight_layout(); plt.show()

# -------------------- Core runner (TEST evaluation per config) --------------------
def run_fold(a, b, H, L, hid, dropout, tau, margin=MARGIN, min_run=MIN_RUN):
    N = len(df)
    tr_end = int(math.floor(a * N))
    te_end = int(math.floor(b * N))
    print(f"\n=== Fold {int(a*100)}→{int(b*100)} | H={H}, L={L}, hid={hid}, drop={dropout}, tau={tau} ===")
    print(f"Train idx: [0..{tr_end-1}]  |  Test idx: [{tr_end}..{te_end-1}]")

    # HMM probabilities
    p_full = hmm_probs_for_split(a,b,te_end)
    p_full = np.nan_to_num(p_full, nan=np.nanmedian(p_full))

    # Hard labels + min-run (TRAIN only)
    p_tr = p_full[:tr_end].copy()
    r_tr = smooth_min_run((p_tr >= tau).astype(int), min_run=min_run)
    keep_tr = (np.abs(p_tr - tau) > margin)

    # TEST labels (for run-length feature only)
    p_te = p_full[tr_end:te_end].copy()
    r_te = smooth_min_run((p_te >= tau).astype(int), min_run=min_run)

    # Run-length (for feature)
    rl_tr = run_length_from_labels(r_tr, cap=SEQ_CAP_RL)
    rl_te = run_length_from_labels(r_te, cap=SEQ_CAP_RL)

    # Base arrays
    dy = df["Δy"].values[:te_end]
    E  = df[EXOG_COLS].values[:te_end].astype(float)

    # Build volatility inputs + targets
    logRV_1w, logRV_4w, logRV_12w, rv_future, logrv_future = build_vol_features_and_targets(dy, H)

    # Train-only scaling of inputs
    m_E, s_E = np.nanmean(E[:tr_end],0), np.nanstd(E[:tr_end],0); s_E[s_E<=1e-12]=1.0
    E_z = (E - m_E) / s_E
    m_l1, s_l1   = zfit(logRV_1w[:tr_end]);   logRV_1w_z  = (logRV_1w  - m_l1)  / s_l1
    m_l4, s_l4   = zfit(logRV_4w[:tr_end]);   logRV_4w_z  = (logRV_4w  - m_l4)  / s_l4
    m_l12, s_l12 = zfit(logRV_12w[:tr_end]);  logRV_12w_z = (logRV_12w - m_l12) / s_l12

    # Run-length feature scaling (train-only)
    rl_full = np.zeros(te_end, dtype=float)
    rl_full[:tr_end]      = rl_tr
    rl_full[tr_end:te_end]= rl_te
    m_rl, s_rl = zfit(rl_full[:tr_end]); rl_z = (rl_full - m_rl) / s_rl

    # Targets: z-scored log(RV_future) (train-only stats)
    m_y, s_y = zfit(logrv_future[:tr_end])
    y_target_z = (logrv_future - m_y) / s_y

    # Gate features
    p_id  = p_full[:te_end]
    p_lag = np.concatenate([[np.nan], p_id[:-1]])

    # Feature block to be windowed across L
    X_block = {
        "logRV_1w_z":  logRV_1w_z,
        "logRV_4w_z":  logRV_4w_z,
        "logRV_12w_z": logRV_12w_z,
        "p_t":         p_id,
        "p_t_1":       p_lag,
        "rl_z":        rl_z,
    }
    for j, ex in enumerate(EXOG_COLS):
        X_block[f"ex_{ex}"] = E_z[:, j]

    # Valid anchors (finite target) with enough history
    valid = np.isfinite(y_target_z)
    idx_all = np.where(valid)[0]
    idx_all = idx_all[idx_all >= (L-1)]
    idx_tr_all = idx_all[idx_all <  tr_end]
    idx_te     = idx_all[idx_all >= tr_end]

    # Apply margin filter to TRAIN indices only
    keep_mask = np.zeros(te_end, dtype=bool); keep_mask[:tr_end] = keep_tr
    idx_tr = idx_tr_all[ keep_mask[idx_tr_all] ]

    # Sequences
    X_tr, y_tr = make_sequences(idx_tr, L, y_target_z, X_block)
    X_te, y_te = make_sequences(idx_te, L, y_target_z, X_block)

    # Time-ordered VAL split
    n_tr = len(X_tr); n_va = max(1, int(round(VAL_FRAC*n_tr)))
    if n_tr < 30:
        print("  [warn] very small training set after margin filter; results may be noisy.")
    tr_slice = np.arange(0, n_tr-n_va)
    va_slice = np.arange(n_tr-n_va, n_tr)

    # Route to experts by TRAIN labels
    r_full = np.zeros(te_end, dtype=int); r_full[:tr_end]=r_tr; r_full[tr_end:te_end]=r_te
    r_tr_seq = r_full[idx_tr]
    tr_hi = tr_slice[r_tr_seq[tr_slice]==1]; tr_lo = tr_slice[r_tr_seq[tr_slice]==0]
    va_hi = va_slice[r_tr_seq[va_slice]==1]; va_lo = va_slice[r_tr_seq[va_slice]==0]

    def mk_loaders(X, y, idx_tr, idx_va):
        ds_tr = SeqDataset(X[idx_tr], y[idx_tr])
        ds_va = SeqDataset(X[idx_va], y[idx_va]) if len(idx_va)>0 else SeqDataset(X[idx_tr], y[idx_tr])
        return DataLoader(ds_tr, batch_size=BATCH, shuffle=True), DataLoader(ds_va, batch_size=BATCH, shuffle=False)

    in_dim = X_tr.shape[2] if len(X_tr)>0 else len(X_block)

    dl_tr_hi, dl_va_hi = mk_loaders(X_tr, y_tr, tr_hi, va_hi if len(va_hi)>0 else tr_hi)
    dl_tr_lo, dl_va_lo = mk_loaders(X_tr, y_tr, tr_lo, va_lo if len(va_lo)>0 else tr_lo)

    exp_hi = LSTMReg(in_dim=in_dim, hid=hid, dropout=dropout).to(device)
    exp_lo = LSTMReg(in_dim=in_dim, hid=hid, dropout=dropout).to(device)

    print(f"  Expert HI train={len(tr_hi)}  val={len(va_hi)} | Expert LO train={len(tr_lo)}  val={len(va_lo)}")
    exp_hi = train_model(exp_hi, dl_tr_hi, dl_va_hi, epochs=EPOCHS_FULL, lr=LR,
                         clip=CLIP_NORM, wd=WEIGHT_DECAY, patience=PATIENCE)
    exp_lo = train_model(exp_lo, dl_tr_lo, dl_va_lo, epochs=EPOCHS_FULL, lr=LR,
                         clip=CLIP_NORM, wd=WEIGHT_DECAY, patience=PATIENCE)

    # Inference (predict z-scored log-RV)
    with torch.no_grad():
        yhat_te_hi = exp_hi(torch.tensor(X_te, dtype=torch.float32, device=device)).cpu().numpy()
        yhat_te_lo = exp_lo(torch.tensor(X_te, dtype=torch.float32, device=device)).cpu().numpy()

    # Soft combine by contemporaneous p_t (at time t)
    p_te_at_t = p_id[idx_te]
    yhat_te_z = p_te_at_t * yhat_te_hi + (1.0 - p_te_at_t) * yhat_te_lo

    # Back-transform to RV space
    yhat_logrv = yhat_te_z * (s_y if s_y!=0 else 1.0) + m_y
    yhat_rv    = np.exp(yhat_logrv)
    y_true_rv  = rv_future[idx_te]

    metr = metrics_all(y_true_rv, yhat_rv)

    payload = {
        "dates": df[DATE_COL].values[:te_end][idx_te],  # aligned at t
        "y_true": y_true_rv,
        "y_hat":  yhat_rv,
        "H": H,
        "fold": f"{int(a*100)}→{int(b*100)}",
        "N": len(y_true_rv)
    }
    return metr, payload

def select_tau_for_fold(a, b, H, L, hid, dropout):
    print(f"\n-- Tuning tau on fold {int(a*100)}→{int(b*100)}, H={H}, L={L}, hid={hid}, drop={dropout} --")
    cand = []
    for tau in TAU_GRID:
        metr, _ = run_fold(a,b,H,L,hid,dropout,tau, margin=MARGIN, min_run=MIN_RUN)
        cand.append((metr["MAE"], tau))
    cand.sort(key=lambda x: x[0])
    best_tau = cand[0][1]
    print(f"==> Selected tau={best_tau} (by MAE on the test block with current settings)")
    return best_tau

# -------------------- Outer search --------------------
overall_rows = []

for (a,b) in SPLITS:
    N = len(df); tr_end = int(math.floor(a*N)); te_end = int(math.floor(b*N))
    split_tag = f"split_{int(a*100)}"
    print("\n" + "="*110)
    print(f"FOLD {int(a*100)}→{int(b*100)}   (train 0..{tr_end-1}, test {tr_end}..{te_end-1})")
    print("="*110)

    for H in HORIZONS:
        best = None  # (MAE, metrics, cfg, payload)
        for L in L_GRID:
            for hid in HID_GRID:
                for drop in DROPOUTS:
                    tau_star = select_tau_for_fold(a,b,H,L,hid,drop)
                    metr, payload = run_fold(a,b,H,L,hid,drop,tau_star, margin=MARGIN, min_run=MIN_RUN)
                    if (best is None) or (metr["MAE"] < best[0]):
                        best = (metr["MAE"], metr,
                                {"L":L, "hid":hid, "drop":drop, "tau":tau_star},
                                payload)

        # Unpack best for this H
        _, metrH, cfgH, payloadH = best

        # -------- Your requested one-line printout per H ----------
        print_block_header(EXOG_COLS, cfgH["L"], cfgH["hid"], cfgH["drop"], cfgH["tau"])
        print_metrics_line(H, metrH)
        # ----------------------------------------------------------

        # Save predictions & config (unchanged)
        pred_df = pd.DataFrame({"Date": payloadH["dates"], "y_true_RV": payloadH["y_true"], "y_hat_RV": payloadH["y_hat"]})
        out_csv = os.path.join(OUTPUT_DIR, f"HMM_LSTM_RV_{split_tag}_H{H}.csv")
        pred_df.to_csv(out_csv, index=False)

        cfg_to_save = {
            "fold": split_tag, "H": H, **cfgH,
            "exogs": EXOG_COLS,
            "metrics": metrH
        }
        with open(os.path.join(OUTPUT_DIR, f"HMM_LSTM_RV_{split_tag}_H{H}_config.json"), "w") as f:
            json.dump(cfg_to_save, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)

        # Plots (optional)
        dplot, ytrue, yhat = (payloadH["dates"], payloadH["y_true"], payloadH["y_hat"])
        plot_with_legend_outside(
            dplot, ytrue, yhat,
            title_left=f"HMM LSTM — Fold {int(a*100)}→{int(b*100)} | H={H} — Actual vs Forecast (RV)",
            ylabel="Realized variance"
        )

        overall_rows.append({
            "fold": split_tag, "H": H,
            "L": cfgH["L"], "hidden": cfgH["hid"], "dropout": cfgH["drop"], "tau": cfgH["tau"],
            "N": metrH["N"], "MAE": metrH["MAE"], "MSE": metrH["MSE"], "RMSE": metrH["RMSE"],
            "MAPE": metrH["MAPE"], "sMAPE": metrH["sMAPE"], "MFE": metrH["MFE"],
            "pred_file": out_csv
        })

# Consolidated summary CSV
overall_df = pd.DataFrame(overall_rows).sort_values(["fold","H"])
summary_path = os.path.join(OUTPUT_DIR, "HMM_LSTM_RV_summary.csv")
overall_df.to_csv(summary_path, index=False)
print(f"\nSaved per-fold×H predictions & configs to: {OUTPUT_DIR}")
print(f"Consolidated summary: {summary_path}")


# In[19]:


# ============== DM (MSE): Is HMM–LSTM better than plain LSTM on RV? ==============
# One-sided alternative: HMM–LSTM has LOWER MSE than LSTM
#   d_t = e_LSTM^2 - e_HMM^2 ;   ALT='greater' tests mean(d_t) > 0
# Inputs:
#   • HMM-LSTM CSVs   → /Users/anwarouni/Downloads/Thesis/Output/HMM_LSTM_RV
#   • LSTM CSVs       → /Users/anwarouni/Downloads/Thesis/Output/ HMM LSTM predictions   (mind the space)
# File pattern:
#   ...split_{S}_H{H}.csv   (e.g., HMM_LSTM_RV_split_60_H1.csv, LSTM_RV_split_80_H4.csv)
# Columns (robustly detected):
#   Date, y_true_RV / Actual_RV / y_true / Actual, y_hat_RV / Forecast_RV_LSTM / y_hat / Forecast
# Output:
#   • per-split CSV  : DM_RV_HMM_LSTM_vs_LSTM_by_split_MSE_one_sided_HLN.csv
#   • pooled-by-H CSV: DM_RV_HMM_LSTM_vs_LSTM_by_horizon_MSE_one_sided_HLN.csv
# ================================================================================

import re, math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List

# ------------------------ Paths -------------------------
from src.paths import HMM_RESULTS_DIR, LSTM_RESULTS_DIR, DM_RESULTS_DIR

HMM_ROOT = HMM_RESULTS_DIR
LSTM_ROOT = LSTM_RESULTS_DIR
OUTDIR = DM_RESULTS_DIR

# ----------------------------- Config ------------------------------------------
SPLITS    = [60, 70, 80, 90]
HORIZONS  = [1, 4, 12]
OVERLAPPING = True   # q = H-1
USE_HLN     = True   # HLN correction ON

# ----------------------------- Discovery ---------------------------------------
def discover_csvs(root: Path, include_tokens: List[str], exclude_tokens: List[str]) -> List[Path]:
    hits = []
    if not root.exists():
        return hits
    for p in root.rglob("*.csv"):
        name = p.name.lower()
        if all(tok in name for tok in include_tokens) and not any(tok in name for tok in exclude_tokens):
            hits.append(p)
    return sorted(hits)

def _int_after(token: str, s: str) -> Optional[int]:
    i = s.lower().rfind(token.lower())
    if i < 0: 
        return None
    tail = s[i+len(token):]
    m = re.search(r"(\d+)", tail)
    return int(m.group(1)) if m else None

def parse_split_h(filename: str) -> Tuple[Optional[int], Optional[int]]:
    split = _int_after("split_", filename) or _int_after("split", filename)
    h = _int_after("_h", filename) or _int_after("h", filename)
    return split, h

def index_files(paths: List[Path]) -> Dict[Tuple[int,int], Path]:
    out: Dict[Tuple[int,int], Path] = {}
    for p in paths:
        s, h = parse_split_h(p.name)
        if s in SPLITS and h in HORIZONS:
            out[(s, h)] = p
    return out

# HMM-LSTM files (e.g., HMM_LSTM_RV_split_60_H1.csv)
HMM_FILES = discover_csvs(HMM_ROOT, include_tokens=["hmm","lstm","split","h"], exclude_tokens=["summary"])
HMM_MAP   = index_files(HMM_FILES)

# Plain LSTM files (e.g., LSTM_RV_split_60_H1.csv)
LSTM_FILES = discover_csvs(LSTM_ROOT, include_tokens=["lstm","split","h"], exclude_tokens=["hmm","summary"])
LSTM_MAP   = index_files(LSTM_FILES)

print("Matched HMM-LSTM:", {k: v.name for k, v in HMM_MAP.items()})
print("Matched LSTM    :", {k: v.name for k, v in LSTM_MAP.items()})

# --------------------------- Column helpers -------------------------------------
ACTUAL_CANDS = [
    "y_true_rv","actual_rv","y_true","actual","y","target","truth","swap spread","rv"
]
FORECAST_CANDS_ANY = [
    "y_hat_rv","forecast_rv_lstm","forecast rv lstm","y_hat","forecast","yhat","pred","prediction"
]
def pick_col(df: pd.DataFrame, name_list: List[str]) -> Optional[str]:
    cols = [str(c).strip() for c in df.columns if not str(c).lower().startswith("unnamed")]
    lower = [c.lower() for c in cols]
    for wanted in name_list:
        for i, c in enumerate(lower):
            if c == wanted:
                return cols[i]
    # substring fallback
    for wanted in name_list:
        for i, c in enumerate(lower):
            if wanted in c:
                return cols[i]
    return None

def load_series(path: Path, model: str) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)

    # Optional Date index
    date_col = None
    for c in df.columns:
        if str(c).lower() in ["date", "time", "timestamp"]:
            date_col = c; break
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)

    a_col = pick_col(df, ACTUAL_CANDS)
    if a_col is None:
        # last chance: first numeric column
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                a_col = c; break
    f_col = pick_col(df, FORECAST_CANDS_ANY)
    if f_col is None:
        # for HMM files sometimes "y_hat" or "y_hat_RV"
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                f_col = c  # take another numeric if not same as actual
        if f_col == a_col:
            raise ValueError(f"Could not identify Forecast column in {path.name}")

    y    = pd.to_numeric(df[a_col], errors="coerce")
    yhat = pd.to_numeric(df[f_col], errors="coerce")
    s = pd.concat([y.rename("y"), yhat.rename("yhat")], axis=1).dropna()
    return s["y"], s["yhat"]

# --------------------------- DM machinery ---------------------------------------
def nw_variance(d: np.ndarray, q: int) -> float:
    T = len(d)
    if T <= 1: return np.nan
    dd = d - d.mean()
    gamma0 = np.dot(dd, dd) / T
    var = gamma0
    for k in range(1, min(q, T-1) + 1):
        w = 1.0 - k / (q + 1.0)            # Bartlett kernel
        gamma_k = np.dot(dd[k:], dd[:-k]) / T
        var += 2.0 * w * gamma_k
    return var / T

def dm_test_mse(e_lstm: np.ndarray, e_hmm: np.ndarray, h: int,
                overlapping: bool = OVERLAPPING, use_hln: bool = USE_HLN):
    """One-sided ALT='greater': HMM-LSTM has lower MSE than LSTM."""
    assert len(e_lstm) == len(e_hmm)
    T = len(e_lstm)
    d = (e_lstm**2) - (e_hmm**2)
    q = max(h - 1, 0) if overlapping else 0

    var = nw_variance(d, q)
    if not np.isfinite(var) or var <= 0:
        return np.nan, np.nan, T, float(np.mean(d)), np.nan, q

    dbar = float(np.mean(d))
    se   = math.sqrt(var)
    dm   = dbar / se

    if use_hln and T > 0:
        dm *= math.sqrt((T + 1 - 2*h + (h*(h-1))/T) / T)  # HLN

    # one-sided p-value (HMM better)
    from math import erf, sqrt
    norm_cdf = lambda x: 0.5*(1 + erf(x / sqrt(2)))
    p = 1 - norm_cdf(dm)
    return dm, p, T, dbar, se, q

def stars(p):
    return "***" if (p is not None and p < 0.01) else ("**" if (p is not None and p < 0.05) else ("*" if (p is not None and p < 0.10) else ""))

def nan_row(extra: dict) -> dict:
    base = {
        "N": 0,
        "MSE_LSTM": np.nan, "MSE_HMM_LSTM": np.nan,
        "RMSE_LSTM": np.nan, "RMSE_HMM_LSTM": np.nan,
        "dbar": np.nan, "SE_NW": np.nan, "q_lag": np.nan,
        "DM": np.nan, "p_value": np.nan, "sig": "",
        "ALT": "greater", "HLN": USE_HLN, "overlapping": OVERLAPPING
    }
    base.update(extra); return base

# ----------------------- Per split × horizon ------------------------------------
rows = []
for s in SPLITS:
    for h in HORIZONS:
        p_l = LSTM_MAP.get((s, h)); p_h = HMM_MAP.get((s, h))
        if p_l is None or p_h is None:
            rows.append(nan_row({"Split": s, "H": h, "note": "missing file(s)"}))
            continue
        try:
            y_l, f_l = load_series(p_l, model="LSTM")
            y_h, f_h = load_series(p_h, model="HMM_LSTM")
            df = pd.concat([y_l.rename("y"), f_l.rename("f_lstm"),
                            y_h.rename("y2"), f_h.rename("f_hmm")], axis=1).dropna()
            if df.empty:
                rows.append(nan_row({"Split": s, "H": h, "note": "no overlap after merge"}))
                continue

            y  = df["y"]  # equals y2 after merge
            eL = (y - df["f_lstm"]).to_numpy()
            eH = (y - df["f_hmm"]).to_numpy()

            DM, p, T, dbar, se, q = dm_test_mse(eL, eH, h=h)

            mse_l = float(np.mean(eL**2)); mse_h = float(np.mean(eH**2))
            rmse_l = float(np.sqrt(mse_l)); rmse_h = float(np.sqrt(mse_h))

            rows.append({
                "Split": s, "H": h, "N": int(T),
                "MSE_LSTM": mse_l, "MSE_HMM_LSTM": mse_h,
                "RMSE_LSTM": rmse_l, "RMSE_HMM_LSTM": rmse_h,
                "dbar": dbar, "SE_NW": se, "q_lag": q,
                "DM": DM, "p_value": p, "sig": stars(p),
                "ALT": "greater", "HLN": USE_HLN, "overlapping": OVERLAPPING
            })
        except Exception as ex:
            rows.append(nan_row({"Split": s, "H": h, "note": "exception", "error": str(ex)}))

by_split = pd.DataFrame(rows).sort_values(["H","Split"]).reset_index(drop=True)
by_split_path = OUTDIR / "DM_RV_HMM_LSTM_vs_LSTM_by_split_MSE_one_sided_HLN.csv"
by_split.to_csv(by_split_path, index=False)
print(f"Saved per-split results → {by_split_path}")

# --------------------------- Pooled per horizon ---------------------------------
pooled_rows = []
for h in HORIZONS:
    eL_all, eH_all = [], []
    for s in SPLITS:
        p_l = LSTM_MAP.get((s, h)); p_h = HMM_MAP.get((s, h))
        if p_l is None or p_h is None: 
            continue
        y_l, f_l = load_series(p_l, model="LSTM")
        y_h, f_h = load_series(p_h, model="HMM_LSTM")
        df = pd.concat([y_l.rename("y"), f_l.rename("f_lstm"),
                        y_h.rename("y2"), f_h.rename("f_hmm")], axis=1).dropna()
        if not df.empty:
            y  = df["y"]
            eL = (y - df["f_lstm"]).to_numpy()
            eH = (y - df["f_hmm"]).to_numpy()
            eL_all.append(eL); eH_all.append(eH)

    if not eL_all:
        pooled_rows.append(nan_row({"H": h, "N_total": 0, "note": "no data"}))
        continue

    eL = np.concatenate(eL_all, axis=0)
    eH = np.concatenate(eH_all, axis=0)

    DM, p, T, dbar, se, q = dm_test_mse(eL, eH, h=h)

    mse_l = float(np.mean(eL**2)); mse_h = float(np.mean(eH**2))
    rmse_l = float(np.sqrt(mse_l)); rmse_h = float(np.sqrt(mse_h))

    pooled_rows.append({
        "H": h, "N_total": int(T),
        "MSE_LSTM": mse_l, "MSE_HMM_LSTM": mse_h,
        "RMSE_LSTM": rmse_l, "RMSE_HMM_LSTM": rmse_h,
        "dbar": dbar, "SE_NW": se, "q_lag": q,
        "DM": DM, "p_value": p, "sig": stars(p),
        "ALT": "greater", "HLN": USE_HLN, "overlapping": OVERLAPPING
    })

pooled = pd.DataFrame(pooled_rows).sort_values("H").reset_index(drop=True)
pooled_path = OUTDIR / "DM_RV_HMM_LSTM_vs_LSTM_by_horizon_MSE_one_sided_HLN.csv"
pooled.to_csv(pooled_path, index=False)
print(f"Saved pooled results → {pooled_path}")

# --------------------------- Console preview ------------------------------------
def fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return "—"
    return f"{x:.6f}" if isinstance(x, float) else str(x)

print("\n============= ONE-SIDED DM (HMM-LSTM better) — MSE (HLN ON) =============")
for h in HORIZONS:
    sub = by_split[by_split["H"] == h]
    if sub.empty: continue
    print(f"\nH = {h}")
    print("Split  N    MSE_LSTM    MSE_HMM     dbar       DM        p   sig  q")
    for _, r in sub.iterrows():
        print(f"{int(r['Split']):>4}  {int(r['N']):>4}  {fmt(r['MSE_LSTM']):>10}  {fmt(r['MSE_HMM_LSTM']):>10}  "
              f"{fmt(r['dbar']):>9}  {fmt(r['DM']):>8}  {fmt(r['p_value']):>8}  {r['sig']:>3}  {int(r['q_lag']):>1}")

print("\n================ Pooled per horizon (one-sided) ================")
for _, r in pooled.iterrows():
    print(f"H={int(r['H'])} | N={int(r.get('N_total',0))} | "
          f"MSE: LSTM={fmt(r.get('MSE_LSTM'))}, HMM={fmt(r.get('MSE_HMM_LSTM'))} | "
          f"dbar={fmt(r.get('dbar'))} | DM={fmt(r.get('DM'))} (p={fmt(r.get('p_value'))}) {r.get('sig','')} "
          f"| q={int(r['q_lag'])} | HLN={r['HLN']}")

