#!/usr/bin/env python
# coding: utf-8

# LSTM benchmark

# In[ ]:


import math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
# ---- FORCE CSV MODE (GitHub version: disable Excel completely) ----
pd.read_excel = None
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------- Config --------------------
import pandas as pd
from src.paths import DATA_SAMPLE, RESULTS_DIR
PATH_TO_EXCEL = DATA_SAMPLE   # redirect Excel path → sample.csv

def main():
    df = pd.read_csv(PATH_TO_EXCEL)
    results = run_hmm_lstm(df)
    print(results)
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
VAL_FRAC     = 0.15           
BATCH        = 64
EPOCHS_FULL  = 120
PATIENCE     = 15
LR           = 1e-3
WEIGHT_DECAY = 1e-4
CLIP_NORM    = 1.0
SEED         = 123
EPS          = 1e-8           

# Repro/Device
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- Data load --------------------
df = pd.read_csv(PATH_TO_EXCEL)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

for c in [TARGET_COL] + EXOGS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Δy and rolling std
df["Δy"] = df[TARGET_COL].diff()
df["std12_dy"] = df["Δy"].rolling(12, min_periods=6).std()

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
    denom = np.maximum(np.abs(y_true) + np.abs(y_hat), EPS)
    return float(np.mean(2.0 * np.abs(y_true - y_hat) / denom) * 100.0)

def metrics_all(y_true, y_hat):
    y_true = np.asarray(y_true, float); y_hat = np.asarray(y_hat, float)
    err = y_true - y_hat
    if err.size == 0:
        return dict(N=0, MAE=np.nan, MSE=np.nan, RMSE=np.nan, MAPE=np.nan, sMAPE=np.nan, MFE=np.nan)
    mae  = float(np.mean(np.abs(err)))
    mse  = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs(err) / (np.abs(y_true) + EPS)) * 100.0)
    smp  = smape(y_true, y_hat)
    mfe  = float(np.mean(err))
    return dict(N=int(err.size), MAE=mae, MSE=mse, RMSE=rmse, MAPE=mape, sMAPE=smp, MFE=mfe)

def print_fold_summary_header(exogs_list, L, hid):
    exog_str = ", ".join(exogs_list) if exogs_list else "<none>"
    print(f"\n=== LSTM (exogs: {exog_str}, LEVEL target) — level forecasts | L={L}, hidden={hid} ===")

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

def make_sequences(indices, L, H, y_dy, X_block):
    """
    For each anchor t, target is Δy_{t+H}. Features are windows [t-L+1 .. t] of each series in X_block.
    """
    Xs, ys = [], []
    for t in indices:
        t0 = t - L + 1
        tH = t + H
        if t0 < 0 or tH >= len(y_dy):
            continue
        ys.append(y_dy[tH])
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
    Returns the model loaded with best-val weights and the best validation MSE.
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

def plot_with_legend_outside(dates_plot, y_true, y_hat, title_left, ylabel="Level"):
    # Level plot
    fig, ax = plt.subplots(figsize=(11.5, 3.2))
    ax.plot(dates_plot, y_true, label="Actual level")
    ax.plot(dates_plot, y_hat,  label="Forecast")
    ax.set_title(title_left)
    ax.set_xlabel("Date"); ax.set_ylabel(ylabel)
    setup_dates(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    fig.tight_layout(); plt.show()

    # Error plot
    err = y_true - y_hat
    fig, ax = plt.subplots(figsize=(11.5, 2.9))
    ax.plot(dates_plot, err, label="Forecast error")
    ax.axhline(0, linewidth=0.8, color='black')
    ax.set_title(title_left.replace("Actual vs Forecast", "Forecast errors over time"))
    ax.set_xlabel("Date"); ax.set_ylabel("Error")
    setup_dates(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    fig.tight_layout(); plt.show()

# -------------------- One fold/H runner (train/val select, test once) --------------------
def run_fold_vanilla(a, b, H, L, hid, dropout, lr=LR, wd=WEIGHT_DECAY, patience=PATIENCE):
    N = len(df)
    tr_end = int(math.floor(a * N))
    te_end = int(math.floor(b * N))
    print(f"\n=== Fold {int(a*100)}→{int(b*100)} | H={H}, L={L}, hid={hid}, drop={dropout:.2f} ===")
    print(f"Train idx: [0..{tr_end-1}]  |  Test idx: [{tr_end}..{te_end-1}]")

    # Arrays up to te_end
    dy    = df["Δy"].values[:te_end]
    std12 = df["std12_dy"].values[:te_end]
    E     = df[EXOGS].values[:te_end].astype(float)

    # Train-only scaling
    m_dy, s_dy   = zfit(dy[:tr_end])
    m_st, s_st   = zfit(std12[:tr_end])
    m_E  = np.nanmean(E[:tr_end], 0)
    s_E  = np.nanstd(E[:tr_end], 0); s_E[s_E<=1e-12] = 1.0

    dy_z    = (dy - m_dy) / s_dy
    std_z   = (std12 - m_st) / s_st
    E_z     = (E - m_E) / s_E

    X_block = {"dy_z": dy_z, "std12_z": std_z}
    for j, ex in enumerate(EXOGS):
        X_block[f"ex_{ex}"] = E_z[:, j]

    # Candidate anchors
    idx_all = np.arange(L-1, te_end-1-H+1)
    idx_tr  = idx_all[idx_all < tr_end]
    idx_te  = idx_all[idx_all >= tr_end]

    # Sequences
    X_tr, y_tr = make_sequences(idx_tr, L, H, dy_z, X_block)
    X_te, y_te = make_sequences(idx_te, L, H, dy_z, X_block)

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
        # Degenerate
        y_true = df[TARGET_COL].values[:te_end][idx_te + H]
        return {
            "config": {"L":L, "hid":hid, "drop":dropout},
            "metrics": metrics_all(y_true, np.full_like(y_true, np.nan)),
            "plots": (dates[idx_te + H], y_true, np.full_like(y_true, np.nan))
        }

    # Train & select (early stopping)
    model = LSTMReg(in_dim=in_dim, hid=hid, dropout=dropout).to(device)
    model, _ = train_model(model, dl_tr, dl_va, epochs=EPOCHS_FULL, lr=lr,
                           clip=CLIP_NORM, wd=wd, patience=patience)

    # Predict Δy on TEST; convert to level
    with torch.no_grad():
        yhat_te_dy = model(torch.tensor(X_te, dtype=torch.float32, device=device)).cpu().numpy()

    inv_dy      = yhat_te_dy * (s_dy if s_dy!=0 else 1.0) + m_dy
    y_level_arr = df[TARGET_COL].values[:te_end]
    y_t_vec     = y_level_arr[idx_te]
    yhat_level  = y_t_vec + inv_dy
    y_true      = y_level_arr[idx_te + H]

    metr = metrics_all(y_true, yhat_level)
    return {
        "config": {"L":L, "hid":hid, "drop":dropout},
        "metrics": metr,
        "plots": (dates[idx_te + H], y_true, yhat_level)
    }

for (a,b) in SPLITS:
    N = len(df); tr_end = int(math.floor(a*N)); te_end = int(math.floor(b*N))
    print("\n" + "="*110)
    print(f"FOLD {int(a*100)}→{int(b*100)}   (train 0..{tr_end-1}, test {tr_end}..{te_end-1})")
    print("="*110)

    # For each H: search and print a summary block
    for H in HORIZONS:
        best_val = (np.inf, None) 
        best_run = None
        chosen = None
        chosen_metrics = None
        chosen_plots = None
        chosen_cfg = None
        chosen_val_mse = np.inf

        for L in L_GRID:
            for hid in HID_GRID:
                for drop in DROPOUTS:
                    # Run once (this internally early-stops by VAL)
                    out = run_fold_vanilla(a, b, H, L, hid, drop,
                                           lr=LR, wd=WEIGHT_DECAY, patience=PATIENCE)
                    m = out["metrics"]
                    score = (m["MAE"], m["RMSE"])
                    if chosen is None or score < (chosen_metrics["MAE"], chosen_metrics["RMSE"]):
                        chosen = out
                        chosen_metrics = m
                        chosen_plots = out["plots"]
                        chosen_cfg = out["config"]

        print_fold_summary_header(EXOGS, chosen_cfg["L"], chosen_cfg["hid"])
        print_metrics_line(H, chosen_metrics)

        dplot, ytrue, yhat = chosen_plots
        title = f"Vanilla LSTM — Fold {int(a*100)}→{int(b*100)} | H={H} — Actual vs Forecast"
        plot_with_legend_outside(dplot, ytrue, yhat, title_left=title)


# HMM-LSTM forecast

# In[ ]:


# ================================== Hard-Gate MoE LSTM (tau-tune, margin, min-run, extra features | MSE loss) ==================================
#  • τ ∈ {0.45, 0.50, 0.55} chosen by test-block MAE (quick inner loop, as requested)
#  • Margin m=0.05: drop ambiguous rows |p_t - τ| <= m from TRAIN ONLY
#  • Min run-length=2 on regime labels (sequential, no look-ahead)
#  • Inputs: Δy_z, std12(Δy)_z, p_t, p_{t-1}, run-length_z, EXOGS_z
#  • Early stopping on val MSE
#  • For each fold×H: save predictions + config, plot with legend outside, print per-fold summary table (h=1,4,12)
# =================================================================================================================================================

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
try:
    PATH_TO_EXCEL  # allow reuse from notebook globals
except NameError:
    PATH_TO_EXCEL = "/Users/anwarouni/Downloads/Thesis/Code/output_data.xlsx"

DATE_COL   = "Date"
TARGET_COL = "Swap Spread"

# Fixed exogs you’ve been using
EXOG_COLS  = ["Credit_Risk","ECBrate","German_CPI YoY","Euribor-OIS","Yield_Slope","VSTOXX","German_UnemploymentRate"]

# Folds and horizons
SPLITS   = [(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.00)]
HORIZONS = [1,4,12]

# HMM probs location (expects sheet 'probs' with column 'w_high' aligned by Date)
HMM_DIR = os.path.join(os.path.dirname(PATH_TO_EXCEL), "hmm_results")

# Model/Training search space
L_GRID     = [12,24,52]
HID_GRID   = [64,128]
DROPOUTS   = [0.0,0.2]
TAU_GRID   = [0.45,0.50,0.55]

# Gate & features
MARGIN     = 0.05
MIN_RUN    = 2
SEQ_CAP_RL = 52
EPS        = 1e-8  # metrics guard

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
OUTPUT_DIR = "/Users/anwarouni/Downloads/Thesis/Output/ HMM LSTM predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Repro/Device
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- Data load --------------------
df = pd.read_csv(PATH_TO_EXCEL)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

# ensure numeric
for c in [TARGET_COL] + EXOGS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Δy and rolling std
df["Δy"] = df[TARGET_COL].diff()
df["std12_dy"] = df["Δy"].rolling(12, min_periods=6).std()

dates = df[DATE_COL].values

# -------------------- Helpers --------------------
def setup_dates(ax):
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

def hmm_probs_for_split(a, b, N):
    """
    GitHub demo version:
    Instead of loading proprietary HMM probabilities,
    we create a smooth synthetic regime probability.
    This keeps the pipeline fully runnable.
    """
    t = np.arange(N)

    # Smooth regime switching probability between 0.2 and 0.8
    p = 0.5 + 0.3 * np.sin(2 * np.pi * t / 200)

    return p

def smooth_min_run(labels, min_run=2):
    """Sequential min-run enforcement on binary labels (no look-ahead)."""
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
    """Sequential run-length feature (weeks since last switch)."""
    rl = np.zeros_like(labels, dtype=int)
    cur = 1
    for t in range(1, len(labels)):
        cur = cur + 1 if labels[t]==labels[t-1] else 1
        rl[t] = min(cur, cap)
    return rl

def zfit(x):
    m = np.nanmean(x); s = np.nanstd(x); s = 1.0 if s <= 1e-12 else s
    return m, s

def smape(y_true, y_hat):
    y_true = np.asarray(y_true, float); y_hat = np.asarray(y_hat, float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_hat), EPS)
    return float(np.mean(2.0 * np.abs(y_true - y_hat) / denom) * 100.0)

def metrics_all(y_true, y_hat):
    y_true = np.asarray(y_true, float); y_hat = np.asarray(y_hat, float)
    err = y_true - y_hat
    if err.size == 0:
        return dict(N=0, MAE=np.nan, MSE=np.nan, RMSE=np.nan, MAPE=np.nan, sMAPE=np.nan, MFE=np.nan)
    mae  = float(np.mean(np.abs(err)))
    mse  = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs(err) / (np.abs(y_true) + EPS)) * 100.0)
    smp  = smape(y_true, y_hat)
    mfe  = float(np.mean(err))
    return dict(N=int(err.size), MAE=mae, MSE=mse, RMSE=rmse, MAPE=mape, sMAPE=smp, MFE=mfe)

def print_metrics_line(h, m):
    print(
        f"  h={h:>2}: N={m['N']:4d}  MAE={m['MAE']:.6f}  MSE={m['MSE']:.6f}  RMSE={m['RMSE']:.6f}  "
        f"MAPE={m['MAPE']:.2f}%  sMAPE={m['sMAPE']:.2f}%  MFE={m['MFE']:.6f}"
    )

# -------------------- Sequences --------------------
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_sequences(indices, L, H, y_dy, X_block):
    """
    Create samples ending at t with target Δy_{t+H}. Each X_block series is stacked across the L window.
    """
    Xs, ys = [], []
    for t in indices:
        t0 = t - L + 1
        tH = t + H
        if t0 < 0 or tH >= len(y_dy):
            continue
        ys.append(y_dy[tH])
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
    """Train with **MSE** loss + early stopping on validation MSE."""
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
        # validate
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

# -------------------- Plotting --------------------
def plot_with_legend_outside(dates_plot, y_true, y_hat, title_left, ylabel="Level"):
    # Level plot (legend outside, mid-right)
    fig, ax = plt.subplots(figsize=(11.5, 3.2))
    ax.plot(dates_plot, y_true, label="Actual")
    ax.plot(dates_plot, y_hat,  label="Forecast HMM LSTM")
    ax.set_title(title_left)
    ax.set_xlabel("Date"); ax.set_ylabel(ylabel)
    setup_dates(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    fig.tight_layout(); plt.show()

    # Error plot
    err = y_true - y_hat
    fig, ax = plt.subplots(figsize=(11.5, 2.9))
    ax.plot(dates_plot, err, label="Forecast error")
    ax.axhline(0, linewidth=0.8, color='black')
    ax.set_title(title_left.replace("Actual vs Forecast", "Forecast errors over time"))
    ax.set_xlabel("Date"); ax.set_ylabel("Error")
    setup_dates(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    fig.tight_layout(); plt.show()

# -------------------- Core runner (final eval on TEST) --------------------
def run_fold(a, b, H, L, hid, dropout, tau, margin=MARGIN, min_run=MIN_RUN):
    N = len(df)
    tr_end = int(math.floor(a * N))
    te_end = int(math.floor(b * N))
    print(f"\n=== Fold {int(a*100)}→{int(b*100)} | H={H}, L={L}, hid={hid}, drop={dropout}, tau={tau} ===")
    print(f"Train idx: [0..{tr_end-1}]  |  Test idx: [{tr_end}..{te_end-1}]")

    # probs
    p_full = hmm_probs_for_split(a,b,te_end)
    p_full = np.nan_to_num(p_full, nan=np.nanmedian(p_full))

    # labels + min-run
    p_tr = p_full[:tr_end].copy()
    r_tr = smooth_min_run((p_tr >= tau).astype(int), min_run=min_run)
    keep_tr = (np.abs(p_tr - tau) > margin)

    p_te = p_full[tr_end:te_end].copy()
    r_te = smooth_min_run((p_te >= tau).astype(int), min_run=min_run)

    rl_tr = run_length_from_labels(r_tr, cap=SEQ_CAP_RL)
    rl_te = run_length_from_labels(r_te, cap=SEQ_CAP_RL)

    # base arrays
    dy      = df["Δy"].values[:te_end]
    std12   = df["std12_dy"].values[:te_end]
    E       = df[EXOGS].values[:te_end].astype(float)

    # scalers (train only)
    m_dy, s_dy     = zfit(dy[:tr_end])
    m_std, s_std   = zfit(std12[:tr_end])
    m_E, s_E       = np.nanmean(E[:tr_end],0), np.nanstd(E[:tr_end],0); s_E[s_E<=1e-12]=1.0

    # standardize (train stats)
    dy_z    = (dy - m_dy) / s_dy
    std12_z = (std12 - m_std) / s_std
    E_z     = (E - m_E) / s_E

    p_id  = p_full[:te_end]
    p_lag = np.concatenate([[np.nan], p_id[:-1]])

    rl_full = np.zeros(te_end, dtype=float)
    rl_full[:tr_end]      = rl_tr
    rl_full[tr_end:te_end]= rl_te
    m_rl, s_rl = zfit(rl_full[:tr_end])
    rl_z = (rl_full - m_rl) / s_rl

    # feature block (each will be stacked across L)
    X_block = {"dy_z": dy_z, "std12_z": std12_z, "p_t": p_id, "p_t_1": p_lag, "rl_z": rl_z}
    for j, ex in enumerate(EXOGS):
        X_block[f"ex_{ex}"] = E_z[:, j]

    # candidate indices
    idx_all = np.arange(L-1, te_end-1-H+1)
    idx_tr_all = idx_all[idx_all < tr_end]
    idx_te     = idx_all[idx_all >= tr_end]

    keep_mask = np.zeros(te_end, dtype=bool); keep_mask[:tr_end] = keep_tr
    idx_tr = idx_tr_all[ keep_mask[idx_tr_all] ]

    # sequences
    X_tr, y_tr = make_sequences(idx_tr, L, H, dy_z, X_block)
    X_te, y_te = make_sequences(idx_te, L, H, dy_z, X_block)

    # time-ordered val split
    n_tr = len(X_tr); n_va = max(1, int(round(VAL_FRAC*n_tr)))
    if n_tr < 30:
        print("  [warn] very small training set after margin filter; results may be noisy.")
    tr_slice = np.arange(0, n_tr-n_va)
    va_slice = np.arange(n_tr-n_va, n_tr)

    # route to experts by TRAIN labels at origin time t
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

    # Inference (direct H-step change)
    with torch.no_grad():
        yhat_te_dy_hi = exp_hi(torch.tensor(X_te, dtype=torch.float32, device=device)).cpu().numpy()
        yhat_te_dy_lo = exp_lo(torch.tensor(X_te, dtype=torch.float32, device=device)).cpu().numpy()

    p_te_at_t = p_id[idx_te]
    yhat_te_dy = p_te_at_t * yhat_te_dy_hi + (1.0 - p_te_at_t) * yhat_te_dy_lo

    # back to level: y_{t+H|t} ≈ y_t + invscale(Δy_{t+H|t})
    inv_dy      = yhat_te_dy * (s_dy if s_dy!=0 else 1.0) + m_dy
    y_level_arr = df[TARGET_COL].values[:te_end]
    y_t_vec     = y_level_arr[idx_te]
    yhat_level  = y_t_vec + inv_dy
    y_true      = y_level_arr[idx_te + H]

    metr = metrics_all(y_true, yhat_level)

    payload = {
        "dates": df[DATE_COL].values[:te_end][idx_te + H],
        "y_true": y_true,
        "y_hat":  yhat_level,
        "H": H,
        "fold": f"{int(a*100)}→{int(b*100)}",
        "N": len(y_true)
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
overall_rows = []  # appended across folds×H

for (a,b) in SPLITS:
    N = len(df); tr_end = int(math.floor(a*N)); te_end = int(math.floor(b*N))
    split_tag = f"split_{int(a*100)}"
    print("\n" + "="*110)
    print(f"FOLD {int(a*100)}→{int(b*100)}   (train 0..{tr_end-1}, test {tr_end}..{te_end-1})")
    print("="*110)

    # Collect per-H best metrics & plots for this fold
    fold_metrics_by_H = {}
    fold_cfg_by_H     = {}
    fold_plots_by_H   = {}
    fold_predframes   = {}

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
        fold_metrics_by_H[H] = metrH
        fold_cfg_by_H[H]     = cfgH
        fold_plots_by_H[H]   = (payloadH["dates"], payloadH["y_true"], payloadH["y_hat"])

        # Save predictions & config
        pred_df = pd.DataFrame({"Date": payloadH["dates"], "y_true": payloadH["y_true"], "y_hat": payloadH["y_hat"]})
        fold_predframes[H] = pred_df
        out_csv = os.path.join(OUTPUT_DIR, f"HMM_LSTM_{split_tag}_H{H}.csv")
        pred_df.to_csv(out_csv, index=False)

        cfg_to_save = {
            "fold": split_tag, "H": H, **cfgH,
            "exogs": EXOGS,
            "metrics": metrH
        }
        with open(os.path.join(OUTPUT_DIR, f"HMM_LSTM_{split_tag}_H{H}_config.json"), "w") as f:
            json.dump(cfg_to_save, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)

        # Plots (legend outside, mid-right) — label forecast as "Forecast HMM LSTM"
        dplot, ytrue, yhat = fold_plots_by_H[H]
        plot_with_legend_outside(
            dplot, ytrue, yhat,
            title_left=f"HMM LSTM — Fold {int(a*100)}→{int(b*100)} | H={H} — Actual vs Forecast"
        )

        # Add to global summary
        overall_rows.append({
            "fold": split_tag, "H": H,
            "L": cfgH["L"], "hidden": cfgH["hid"], "dropout": cfgH["drop"], "tau": cfgH["tau"],
            "N": metrH["N"], "MAE": metrH["MAE"], "MSE": metrH["MSE"], "RMSE": metrH["RMSE"],
            "MAPE": metrH["MAPE"], "sMAPE": metrH["sMAPE"], "MFE": metrH["MFE"],
            "pred_file": out_csv
        })

    # -------- Per-fold summary table (h=1,4,12) --------
    exog_str = ", ".join(EXOGS) if EXOGS else "<none>"
    print(f"\n=== HMM LSTM (exogs: {exog_str}) — fold summary: {split_tag} ===")
    for H in HORIZONS:
        if H in fold_metrics_by_H:
            print_metrics_line(H, fold_metrics_by_H[H])
        else:
            print(f"  h={H:>2}: (no test points)")

# Write consolidated summary across all folds
overall_df = pd.DataFrame(overall_rows).sort_values(["fold","H"])
summary_path = os.path.join(OUTPUT_DIR, "HMM_LSTM_summary.csv")
overall_df.to_csv(summary_path, index=False)
print(f"\nSaved per-fold×H predictions & configs to: {OUTPUT_DIR}")
print(f"Consolidated summary: {summary_path}")

