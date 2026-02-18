#!/usr/bin/env python
# coding: utf-8

# TRADE STRATE 10% TOP 10% BOTTOM

# In[ ]:


import os, re, glob, warnings
from pathlib import Path
from typing import Tuple, Optional
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from src.paths import DATA_SAMPLE, RESULTS_DIR

# -------------------- Config --------------------
TOPQ = 0.10  # 10% tails (top and bottom)
TRADE_XL_PATH = "/Users/anwarouni/Downloads/Thesis/Data/TradeData.xlsx"

MODELS = [
    ("LSTM",     "/Users/anwarouni/Downloads/Thesis/Output/LSTM predictions",
                 "/Users/anwarouni/Downloads/Thesis/Output/Trading Backtest (LSTM q10 flip)"),
    ("HMM_LSTM", "/Users/anwarouni/Downloads/Thesis/Output/HMM_LSTM_predictionsTEST_RS1",
                 "/Users/anwarouni/Downloads/Thesis/Output/Trading Backtest (HMM_LSTM q10 flip)"),
]

# -------------------- Helpers --------------------
def load_trade_data(xl_path: str) -> pd.DataFrame:
    path = xl_path
    if (not os.path.exists(path)) and xl_path.endswith(".xslx"):
        alt = xl_path[:-5] + "xlsx"
        if os.path.exists(alt):
            path = alt
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trade data not found at: {xl_path}")

    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception:
        df = pd.read_excel(path)

    cols = {c.lower().strip(): c for c in df.columns}
    date_col = next((cols[k] for k in cols if k in ("date","dates")), None)
    swap_col = next((cols[k] for k in cols if k in ("10yswap","swap","swap10","eur_swap_10y")), None)
    bund_col = next((cols[k] for k in cols if k in ("10ybund","bund","bund10","de_bund_10y","10yboon","10yboond","10yboonD")), None)
    if not date_col or not swap_col or not bund_col:
        raise ValueError(f"Expected columns like Date, 10Yswap, 10Ybund. Found: {list(df.columns)}")

    df = df[[date_col, swap_col, bund_col]].copy()
    df.rename(columns={date_col:"date", swap_col:"swap_10y", bund_col:"bund_10y"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df["spread"] = df["swap_10y"] - df["bund_10y"]
    df.set_index("date", inplace=True)
    return df

def detect_cols(pred: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str]]:
    lower = {c.lower(): c for c in pred.columns}
    date_col = next((lower[k] for k in lower if k in ("date","ds","timestamp","time")), None)
    if date_col is None:
        raise ValueError(f"No date-like column in predictions: {list(pred.columns)}")
    pred_cands = ["y_pred","yhat","y_hat","pred","forecast","yhat_level","y_pred_level"]
    y_pred_col = next((lower[k] for k in lower if k in pred_cands), None)
    true_cands = ["y_true","y","y_level","target","actual","truth"]
    y_true_col = next((lower[k] for k in lower if k in true_cands), None)
    return date_col, y_true_col, y_pred_col

def horizon_from_filename(fname: str) -> Optional[int]:
    m = re.search(r"_H(1|4|12)\.csv$", fname)
    return int(m.group(1)) if m else None

def split_from_filename(fname: str) -> Optional[int]:
    m = re.search(r"_split__?(\d+)_", fname)
    return int(m.group(1)) if m else None

def pick_quantiles(df_sig: pd.DataFrame, col: str, q: float) -> pd.DataFrame:
    """Select top/bottom q by 'col' and set 'signal' {+1 long, -1 short}."""
    lo_thr = df_sig[col].quantile(q)
    hi_thr = df_sig[col].quantile(1 - q)
    sel = df_sig[(df_sig[col] <= lo_thr) | (df_sig[col] >= hi_thr)].copy()
    sel["signal"] = np.where(sel[col] >= hi_thr, +1, -1)
    return sel

# -------- Exit rule: opposite signal (sign flip) --------
def find_exit_on_flip(pred_series: pd.Series, entry_idx: int, direction: int) -> Optional[int]:
    """
    Walk forward from entry_idx. For a long (+1), exit when pred flips negative.
    For a short (-1), exit when pred flips positive. Returns index position or None.
    """
    for i in range(entry_idx + 1, len(pred_series)):
        val = pred_series.iloc[i]
        if direction == +1 and val < 0:
            return i
        if direction == -1 and val > 0:
            return i
    return None

def backtest_one(pred_csv: str, trade_df: pd.DataFrame, topq: float) -> pd.DataFrame:
    H = horizon_from_filename(pred_csv)
    split = split_from_filename(pred_csv)

    pred = pd.read_csv(pred_csv)
    date_col, y_true_col, y_pred_col = detect_cols(pred)
    pred[date_col] = pd.to_datetime(pred[date_col])
    pred.sort_values(by=date_col, inplace=True)

    # Attach true spread level if missing
    if y_true_col is None:
        pred = pred.merge(trade_df["spread"].rename("y_true"),
                          left_on=date_col, right_index=True, how="left")
        y_true_col = "y_true"

    # Find predicted level column
    if y_pred_col is None:
        for c in [f"yhat_H{H}", f"y_pred_H{H}", "yhat", "y_pred", "forecast"]:
            if c in pred.columns:
                y_pred_col = c
                break
        if y_pred_col is None:
            raise ValueError(f"No prediction column found in {pred_csv}. Columns={list(pred.columns)}")

    # Predicted delta = predicted level - current level at prediction time
    pred["pred_delta"] = pred[y_pred_col] - pred[y_true_col]

    # Keep only dates present in trading data
    pred = pred[pred[date_col].isin(trade_df.index)]
    if pred.empty:
        return pd.DataFrame()

    # Select top/bottom quantiles at entry
    sig = pick_quantiles(pred[[date_col, "pred_delta"]].copy(), "pred_delta", q=topq)

    # --- Align predictions to trade_df index safely (avoid 'date' ambiguity) ---
    pred_aligned = (
        pd.DataFrame({"date_idx": trade_df.index})
        .merge(pred[[date_col, "pred_delta"]], left_on="date_idx", right_on=date_col, how="left")
        .set_index("date_idx")["pred_delta"]
        .ffill()    # carry the latest forecast forward until the next update
    )

    # Build entries and find flip exits
    entries = sig[date_col].values
    idx_pos = trade_df.index.get_indexer(entries)  # integer positions in trade_df index

    exits_idx, reasons = [], []
    for pos, direction in zip(idx_pos, sig["signal"].values):
        if pos == -1:
            exits_idx.append(pd.NaT)
            reasons.append("no_entry_match")
            continue
        exit_pos = find_exit_on_flip(pred_aligned, pos, int(direction))
        if exit_pos is None or exit_pos >= len(trade_df.index):
            exits_idx.append(pd.NaT)
            reasons.append("no_exit")
        else:
            exits_idx.append(trade_df.index[exit_pos])
            reasons.append("flip_exit")

    # Assemble trades
    trades = pd.DataFrame({
        "entry_date": entries,
        "exit_date": exits_idx,
        "signal": sig["signal"].values,
        "pred_delta_at_entry": sig["pred_delta"].values,
        "exit_reason": reasons
    }).dropna(subset=["exit_date"]).copy()

    # P&L
    entry_spread = trade_df.loc[trades["entry_date"], "spread"].values
    exit_spread  = trade_df.loc[trades["exit_date"],  "spread"].values
    trades["realized_delta"] = exit_spread - entry_spread
    trades["pnl_units"] = trades["signal"] * trades["realized_delta"]

    # Meta
    trades["horizon_w"] = H
    trades["split_pct"] = split
    trades["src_file"] = os.path.basename(pred_csv)
    return trades

def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    grp = trades.groupby(["horizon_w"]).agg(
        n_trades=("pnl_units","count"),
        hit_rate=("pnl_units", lambda x: np.mean(x>0)),
        avg_pnl=("pnl_units","mean"),
        std_pnl=("pnl_units", lambda x: float(np.std(x, ddof=1)) if len(x)>1 else 0.0),
        total_pnl=("pnl_units","sum"),
        median_pnl=("pnl_units","median")
    ).reset_index()
    grp["sharpe_like"] = grp.apply(
        lambda r: (r["avg_pnl"] / r["std_pnl"]) if r["std_pnl"] not in (0.0, np.nan) else np.nan,
        axis=1
    )
    return grp

def run_model(model_name: str, pred_dir: str, out_dir: str, trade_df: pd.DataFrame, topq: float):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csvs = sorted(glob.glob(os.path.join(pred_dir, "*.csv")))
    csvs = [c for c in csvs if re.search(r"_H(1|4|12)\.csv$", c)]
    if not csvs:
        print(f"[{model_name}] No prediction CSVs found in {pred_dir}")
        return None, None

    all_trades = []
    for fp in csvs:
        try:
            trades = backtest_one(fp, trade_df, topq=topq)
            if not trades.empty:
                all_trades.append(trades)
                out_name = os.path.splitext(os.path.basename(fp))[0] + f"_trades_q{int(topq*100):02d}_flip.csv"
                trades.to_csv(os.path.join(out_dir, out_name), index=False)
                print(f"[{model_name}] {out_name}: {len(trades)} trades")
            else:
                print(f"[{model_name}] [SKIP] No trades for {os.path.basename(fp)}")
        except Exception as e:
            print(f"[{model_name}] [ERROR] {os.path.basename(fp)} -> {e}")

    if not all_trades:
        print(f"[{model_name}] No trades aggregated.")
        return None, None

    all_trades_df = pd.concat(all_trades, ignore_index=True)
    all_trades_df.to_csv(os.path.join(out_dir, f"ALL_trades_q{int(topq*100):02d}_flip.csv"), index=False)

    summary_h = summarize_trades(all_trades_df)
    summary_h.to_csv(os.path.join(out_dir, f"summary_by_horizon_q{int(topq*100):02d}_flip.csv"), index=False)

    pnl = all_trades_df["pnl_units"]
    overall = pd.DataFrame([{
        "n_trades": int(len(pnl)),
        "hit_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "avg_pnl": float(pnl.mean()) if len(pnl) else np.nan,
        "std_pnl": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
        "total_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "median_pnl": float(pnl.median()) if len(pnl) else np.nan,
        "sharpe_like": (pnl.mean() / pnl.std(ddof=1)) if len(pnl) > 1 else np.nan
    }])
    overall.to_csv(os.path.join(out_dir, f"summary_overall_q{int(topq*100):02d}_flip.csv"), index=False)

    # Print nicely
    print(f"\n=== {model_name} — Summary by horizon (q={topq*100:.1f}%, flip-exit) ===")
    print(summary_h.to_string(index=False))
    print(f"\n=== {model_name} — Overall (q={topq*100:.1f}%, flip-exit) ===")
    print(overall.to_string(index=False))
    print("\n" + "-"*90 + "\n")

    return summary_h, overall

# -------------------- Run both models --------------------
def main():
    trade_df = load_trade_data(TRADE_XL_PATH)
    for model_name, pred_dir, out_dir in MODELS:
        run_model(model_name, pred_dir, out_dir, trade_df, TOPQ)

if __name__ == "__main__":
    main()


# In[ ]:


import os
import math
import numpy as np
import pandas as pd
from typing import Tuple

# ---- Where your flip-exit outputs live ----
ROOT = "/Users/anwarouni/Downloads/Thesis/Output"
LSTM_DIR = os.path.join(ROOT, "Trading Backtest (LSTM q10 flip)")
HMM_DIR  = os.path.join(ROOT, "Trading Backtest (HMM_LSTM q10 flip)")

LSTM_ALL = os.path.join(LSTM_DIR, "ALL_trades_q10_flip.csv")
HMM_ALL  = os.path.join(HMM_DIR,  "ALL_trades_q10_flip.csv")

# ---------- Helpers ----------
def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["entry_date","exit_date"])
    # Ensure required columns exist
    need = {"pnl_units","horizon_w"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing columns in {path}. Need {need}, got {df.columns.tolist()}")
    df = df.sort_values("exit_date").reset_index(drop=True)
    df["win"] = (df["pnl_units"] > 0).astype(int)
    return df

def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# ----- Hit rate: two-proportion z-test -----
def hitrate_test(hits1: int, n1: int, hits2: int, n2: int) -> Tuple[float,float,float]:
    """Returns (diff = p1-p2, z, p_two_sided)"""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")
    p1 = hits1 / n1
    p2 = hits2 / n2
    p_pool = (hits1 + hits2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se if se > 0 else float("inf")
    p = 2 * (1 - norm_cdf(abs(z)))
    return (p1 - p2), z, p

# ----- Sharpe ratio: Jobson–Korkie with Memmel correction -----
def sharpe_jk_memmel(r1: np.ndarray, r2: np.ndarray) -> dict:
    """
    r1, r2: aligned per-trade P&L (or returns) arrays (same length).
    Returns dict with SR1, SR2, diff, z, p.
    Note: This is the same functional form we've been using for consistency.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    if len(r1) != len(r2):
        raise ValueError("Sharpe test requires same-length aligned series.")
    n = len(r1)
    if n < 2:
        return {"SR1": np.nan, "SR2": np.nan, "diff": np.nan, "z": np.nan, "p": np.nan}

    mu1, mu2 = r1.mean(), r2.mean()
    s1, s2   = r1.std(ddof=1), r2.std(ddof=1)
    SR1, SR2 = mu1 / s1 if s1>0 else np.nan, mu2 / s2 if s2>0 else np.nan
    cov12    = np.cov(r1, r2, ddof=1)[0,1]

    # Memmel-corrected variance of SR difference (simple, consistent form used before)
    var_d = (1.0 / n) * ( 2 * (1 - SR1**2) * (1 - SR2**2) + 2 * (cov12 / (s1 * s2)) )
    var_d = max(var_d, 1e-12)
    z = (SR1 - SR2) / math.sqrt(var_d)
    p = 2 * (1 - norm_cdf(abs(z)))
    return {"SR1": SR1, "SR2": SR2, "diff": SR1 - SR2, "z": z, "p": p}

def label_sig(name1: str, name2: str, diff: float, p: float, higher_is_better=True) -> str:
    if math.isnan(diff) or math.isnan(p):
        return "n/a"
    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    if diff > 0:
        better = f"{name1} > {name2}"
    elif diff < 0:
        better = f"{name2} > {name1}"
    else:
        better = f"{name1} = {name2}"
    verdict = "significant" if p < 0.05 else "ns"
    return f"{better} ({verdict}, p={p:.4g}) {stars}"

def analyze_block(df1: pd.DataFrame, df2: pd.DataFrame, mask: pd.Series, label: str, name1="LSTM", name2="HMM-LSTM"):
    a = df1[mask]
    b = df2[mask]
    # Hit-rate test
    hits1, n1 = int(a["win"].sum()), int(len(a))
    hits2, n2 = int(b["win"].sum()), int(len(b))
    diff_hr, z_hr, p_hr = hitrate_test(hits1, n1, hits2, n2)

    # Sharpe test: align by exit_date positionally after sorting (both already sorted)
    # We take min length to avoid mismatch (should be equal in your outputs).
    m = min(len(a), len(b))
    r1 = a["pnl_units"].to_numpy()[:m]
    r2 = b["pnl_units"].to_numpy()[:m]
    sr = sharpe_jk_memmel(r1, r2)

    print(f"\n===== {label} =====")
    print(f"Counts: {name1} n={n1}, {name2} n={n2}")
    # Hit rate block
    hr1 = hits1 / n1 if n1 else np.nan
    hr2 = hits2 / n2 if n2 else np.nan
    print(f"Hit rate: {name1}={hr1:.4f}, {name2}={hr2:.4f}, diff={diff_hr:.4f}, z={z_hr:.3f}, p={p_hr:.4g}")
    print(" → " + label_sig(name1, name2, diff_hr, p_hr))
    # Sharpe block
    print(f"Sharpe:  {name1}={sr['SR1']:.6f}, {name2}={sr['SR2']:.6f}, diff={sr['diff']:.6f}, z={sr['z']:.3f}, p={sr['p']:.4g}")
    print(" → " + label_sig(name1, name2, sr["diff"], sr["p"]))

def main():
    lstm = load_trades(LSTM_ALL)
    hmm  = load_trades(HMM_ALL)

    # OVERALL
    analyze_block(lstm, hmm, mask=pd.Series([True]*len(lstm)), label="OVERALL (flip-exit, q=10%)")

    # PER HORIZON
    for H in [1, 4, 12]:
        mask = (lstm["horizon_w"] == H)  # both dfs have same horizons counts in your output
        analyze_block(lstm, hmm, mask=mask, label=f"H= {H} (flip-exit, q=10%)")

if __name__ == "__main__":
    main()


# In[ ]:


import os, re, glob, warnings
from pathlib import Path
from typing import Tuple, Optional
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# -------------------- Config --------------------
TOPQ = 0.10  # 10% tails (top and bottom)
TRADE_XL_PATH = "/Users/anwarouni/Downloads/Thesis/Data/TradeData.xlsx"

MODELS = [
    ("ARP", "/Users/anwarouni/Downloads/Thesis/Output/AR(p) predictions",
            "/Users/anwarouni/Downloads/Thesis/Output/Trading Backtest (ARP q10 flip)"),
    ("ARX", "/Users/anwarouni/Downloads/Thesis/Output/AR-X predictions",
            "/Users/anwarouni/Downloads/Thesis/Output/Trading Backtest (ARX q10 flip)"),
]

# -------------------- Helpers --------------------
def load_trade_data(xl_path: str) -> pd.DataFrame:
    path = xl_path
    if (not os.path.exists(path)) and xl_path.endswith(".xslx"):
        alt = xl_path[:-5] + "xlsx"
        if os.path.exists(alt): path = alt
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trade data not found at: {xl_path}")

    try: df = pd.read_excel(path, engine="openpyxl")
    except Exception: df = pd.read_excel(path)

    cols = {c.lower().strip(): c for c in df.columns}
    date_col = next((cols[k] for k in cols if k in ("date","dates")), None)
    swap_col = next((cols[k] for k in cols if k in ("10yswap","swap","swap10","eur_swap_10y")), None)
    bund_col = next((cols[k] for k in cols if k in ("10ybund","bund","bund10","de_bund_10y","10yboon","10yboond","10yboonD")), None)
    if not date_col or not swap_col or not bund_col:
        raise ValueError(f"Expected columns like Date, 10Yswap, 10Ybund. Found: {list(df.columns)}")

    df = df[[date_col, swap_col, bund_col]].copy()
    df.rename(columns={date_col:"date", swap_col:"swap_10y", bund_col:"bund_10y"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df["spread"] = df["swap_10y"] - df["bund_10y"]
    df.set_index("date", inplace=True)
    return df

def detect_cols(pred: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str]]:
    lower = {c.lower(): c for c in pred.columns}
    date_col = next((lower[k] for k in lower if k in ("date","ds","timestamp","time")), None)
    if date_col is None: raise ValueError(f"No date-like column in predictions: {list(pred.columns)}")
    pred_cands = ["y_pred","yhat","y_hat","pred","forecast","yhat_level","y_pred_level"]
    y_pred_col = next((lower[k] for k in lower if k in pred_cands), None)
    true_cands = ["y_true","y","y_level","target","actual","truth"]
    y_true_col = next((lower[k] for k in lower if k in true_cands), None)
    return date_col, y_true_col, y_pred_col

def horizon_from_filename(fname: str) -> Optional[int]:
    # Accept ..._H1.csv or ..._h1.csv
    m = re.search(r"_[Hh](1|4|12)\.csv$", os.path.basename(fname))
    return int(m.group(1)) if m else None

def split_from_filename(fname: str) -> Optional[int]:
    # Accept ..._split_60_... or ..._split__60_...
    m = re.search(r"_split_+(\d+)_", os.path.basename(fname))
    return int(m.group(1)) if m else None

def find_pred_col(pred: pd.DataFrame, H: Optional[int]) -> Optional[str]:
    cols = list(pred.columns)
    # try horizon-specific first (upper & lower 'h')
    horizon_candidates = []
    if H is not None:
        horizon_candidates += [f"yhat_H{H}", f"y_pred_H{H}", f"forecast_H{H}",
                               f"yhat_h{H}", f"y_pred_h{H}", f"forecast_h{H}"]
    generic = ["yhat","y_pred","forecast","pred","prediction"]
    for c in horizon_candidates + generic:
        if c in cols: return c
        # loose match
        for col in cols:
            if re.fullmatch(c, col, flags=re.IGNORECASE):
                return col
    return None

def pick_quantiles(df_sig: pd.DataFrame, col: str, q: float) -> pd.DataFrame:
    lo_thr = df_sig[col].quantile(q); hi_thr = df_sig[col].quantile(1 - q)
    sel = df_sig[(df_sig[col] <= lo_thr) | (df_sig[col] >= hi_thr)].copy()
    sel["signal"] = np.where(sel[col] >= hi_thr, +1, -1)
    return sel

# -------- Exit rule: opposite signal --------
def find_exit_on_flip(pred_series: pd.Series, entry_idx: int, direction: int) -> Optional[int]:
    for i in range(entry_idx + 1, len(pred_series)):
        val = pred_series.iloc[i]
        if direction == +1 and val < 0: return i
        if direction == -1 and val > 0: return i
    return None

def backtest_one(pred_csv: str, trade_df: pd.DataFrame, topq: float) -> pd.DataFrame:
    H = horizon_from_filename(pred_csv)
    split = split_from_filename(pred_csv)

    if H is None:
        # Skip non-horizon files (e.g., "arx_all_predictions.csv")
        # print(f"[SKIP] Could not parse horizon from {os.path.basename(pred_csv)}")
        return pd.DataFrame()

    pred = pd.read_csv(pred_csv)
    date_col, y_true_col, y_pred_col = detect_cols(pred)
    pred[date_col] = pd.to_datetime(pred[date_col]); pred.sort_values(by=date_col, inplace=True)

    if y_true_col is None:
        pred = pred.merge(trade_df["spread"].rename("y_true"), left_on=date_col, right_index=True, how="left")
        y_true_col = "y_true"

    if y_pred_col is None:
        y_pred_col = find_pred_col(pred, H)
        if y_pred_col is None:
            raise ValueError(f"No prediction column found in {os.path.basename(pred_csv)}. Columns={list(pred.columns)}")

    pred["pred_delta"] = pred[y_pred_col] - pred[y_true_col]
    pred = pred[pred[date_col].isin(trade_df.index)]
    if pred.empty: return pd.DataFrame()

    sig = pick_quantiles(pred[[date_col, "pred_delta"]].copy(), "pred_delta", q=topq)

    # Align to trading calendar and ffill predictions
    pred_aligned = (
        pd.DataFrame({"date_idx": trade_df.index})
        .merge(pred[[date_col, "pred_delta"]], left_on="date_idx", right_on=date_col, how="left")
        .set_index("date_idx")["pred_delta"]
        .ffill()
    )

    entries = sig[date_col].values
    idx_pos = trade_df.index.get_indexer(entries)

    exits_idx, reasons = [], []
    for pos, direction in zip(idx_pos, sig["signal"].values):
        if pos == -1:
            exits_idx.append(pd.NaT); reasons.append("no_entry_match"); continue
        exit_pos = find_exit_on_flip(pred_aligned, pos, int(direction))
        if exit_pos is None or exit_pos >= len(trade_df.index):
            exits_idx.append(pd.NaT); reasons.append("no_exit")
        else:
            exits_idx.append(trade_df.index[exit_pos]); reasons.append("flip_exit")

    trades = pd.DataFrame({
        "entry_date": entries,
        "exit_date": exits_idx,
        "signal": sig["signal"].values,
        "pred_delta_at_entry": sig["pred_delta"].values,
        "exit_reason": reasons
    }).dropna(subset=["exit_date"]).copy()

    entry_spread = trade_df.loc[trades["entry_date"], "spread"].values
    exit_spread  = trade_df.loc[trades["exit_date"],  "spread"].values
    trades["realized_delta"] = exit_spread - entry_spread
    trades["pnl_units"] = trades["signal"] * trades["realized_delta"]

    trades["horizon_w"] = H
    trades["split_pct"] = split
    trades["src_file"] = os.path.basename(pred_csv)
    return trades

def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty: return pd.DataFrame()
    grp = trades.groupby(["horizon_w"]).agg(
        n_trades=("pnl_units","count"),
        hit_rate=("pnl_units", lambda x: np.mean(x>0)),
        avg_pnl=("pnl_units","mean"),
        std_pnl=("pnl_units", lambda x: float(np.std(x, ddof=1)) if len(x)>1 else 0.0),
        total_pnl=("pnl_units","sum"),
        median_pnl=("pnl_units","median")
    ).reset_index()
    grp["sharpe_like"] = grp.apply(
        lambda r: (r["avg_pnl"] / r["std_pnl"]) if r["std_pnl"] not in (0.0, np.nan) else np.nan, axis=1
    )
    return grp

def run_model(model_name: str, pred_dir: str, out_dir: str, trade_df: pd.DataFrame, topq: float):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Grab all CSVs; let backtest_one decide which to skip based on horizon
    csvs = sorted(glob.glob(os.path.join(pred_dir, "*.csv")))
    if not csvs:
        print(f"[{model_name}] No .csv files found in {pred_dir}")
        return None, None

    all_trades = []
    for fp in csvs:
        try:
            trades = backtest_one(fp, trade_df, topq=topq)
            if not trades.empty:
                all_trades.append(trades)
                out_name = os.path.splitext(os.path.basename(fp))[0] + f"_trades_q{int(topq*100):02d}_flip.csv"
                trades.to_csv(os.path.join(out_dir, out_name), index=False)
                print(f"[{model_name}] {out_name}: {len(trades)} trades")
        except Exception as e:
            print(f"[{model_name}] [ERROR] {os.path.basename(fp)} -> {e}")

    if not all_trades:
        print(f"[{model_name}] No trades aggregated in {pred_dir}.")
        return None, None

    all_trades_df = pd.concat(all_trades, ignore_index=True)
    all_trades_df.to_csv(os.path.join(out_dir, f"ALL_trades_q{int(topq*100):02d}_flip.csv"), index=False)

    summary_h = summarize_trades(all_trades_df)
    summary_h.to_csv(os.path.join(out_dir, f"summary_by_horizon_q{int(topq*100):02d}_flip.csv"), index=False)

    pnl = all_trades_df["pnl_units"]
    overall = pd.DataFrame([{
        "n_trades": int(len(pnl)),
        "hit_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "avg_pnl": float(pnl.mean()) if len(pnl) else np.nan,
        "std_pnl": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
        "total_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "median_pnl": float(pnl.median()) if len(pnl) else np.nan,
        "sharpe_like": (pnl.mean() / pnl.std(ddof=1)) if len(pnl) > 1 and pnl.std(ddof=1) != 0 else np.nan
    }])
    overall.to_csv(os.path.join(out_dir, f"summary_overall_q{int(topq*100):02d}_flip.csv"), index=False)

    # Print nicely
    print(f"\n=== {model_name} — Summary by horizon (q={topq*100:.1f}%, flip-exit) ===")
    print(summary_h.to_string(index=False))
    print(f"\n=== {model_name} — Overall (q={topq*100:.1f}%, flip-exit) ===")
    print(overall.to_string(index=False))
    print("\n" + "-"*90 + "\n")

    return summary_h, overall

# -------------------- Run both models --------------------
def main():
    trade_df = load_trade_data(TRADE_XL_PATH)
    for model_name, pred_dir, out_dir in MODELS:
        run_model(model_name, pred_dir, out_dir, trade_df, TOPQ)

if __name__ == "__main__":
    main()


# In[ ]:


import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, Optional

# ---- Where your flip-exit outputs live ----
ROOT     = "/Users/anwarouni/Downloads/Thesis/Output"
HMM_DIR  = os.path.join(ROOT, "Trading Backtest (HMM_LSTM q10 flip)")
ARP_DIR  = os.path.join(ROOT, "Trading Backtest (ARP q10 flip)")
ARX_DIR  = os.path.join(ROOT, "Trading Backtest (ARX q10 flip)")

HMM_ALL  = os.path.join(HMM_DIR, "ALL_trades_q10_flip.csv")
ARP_ALL  = os.path.join(ARP_DIR, "ALL_trades_q10_flip.csv")
ARX_ALL  = os.path.join(ARX_DIR, "ALL_trades_q10_flip.csv")

# ---------- Helpers ----------
def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["entry_date","exit_date"])
    need = {"pnl_units","horizon_w"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing columns in {path}. Need {need}, got {df.columns.tolist()}")
    df = df.sort_values("exit_date").reset_index(drop=True)
    df["win"] = (df["pnl_units"] > 0).astype(int)
    return df

def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# ----- Hit rate: two-proportion z-test (independent samples) -----
def hitrate_test(hits1: int, n1: int, hits2: int, n2: int) -> Tuple[float,float,float]:
    """Returns (diff = p1-p2, z, p_two_sided)"""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")
    p1 = hits1 / n1
    p2 = hits2 / n2
    p_pool = (hits1 + hits2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se if se > 0 else float("inf")
    p = 2 * (1 - norm_cdf(abs(z)))
    return (p1 - p2), z, p

# ----- Sharpe ratio: Jobson–Korkie with Memmel correction -----
def sharpe_jk_memmel(r1: np.ndarray, r2: np.ndarray) -> dict:
    """
    r1, r2: aligned per-trade P&L arrays (same length after trimming to min length).
    Returns dict with SR1, SR2, diff, z, p.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    n = len(r1)
    if n < 2 or len(r2) != n:
        return {"SR1": np.nan, "SR2": np.nan, "diff": np.nan, "z": np.nan, "p": np.nan}

    mu1, mu2 = r1.mean(), r2.mean()
    s1, s2   = r1.std(ddof=1), r2.std(ddof=1)
    SR1, SR2 = (mu1 / s1 if s1 > 0 else np.nan), (mu2 / s2 if s2 > 0 else np.nan)
    cov12    = np.cov(r1, r2, ddof=1)[0,1]

    # Memmel-corrected variance of SR difference (compact form used previously)
    var_d = (1.0 / n) * ( 2 * (1 - SR1**2) * (1 - SR2**2) + 2 * (cov12 / (s1 * s2)) )
    var_d = max(var_d, 1e-12)
    z = (SR1 - SR2) / math.sqrt(var_d)
    p = 2 * (1 - norm_cdf(abs(z)))
    return {"SR1": SR1, "SR2": SR2, "diff": SR1 - SR2, "z": z, "p": p}

def label_sig(name1: str, name2: str, diff: float, p: float) -> str:
    if math.isnan(diff) or math.isnan(p):
        return "n/a"
    stars = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 5e-2 else ""))
    if diff > 0:
        better = f"{name1} > {name2}"
    elif diff < 0:
        better = f"{name2} > {name1}"
    else:
        better = f"{name1} = {name2}"
    verdict = "significant" if p < 0.05 else "ns"
    return f"{better} ({verdict}, p={p:.4g}) {stars}"

def analyze_block(df1: pd.DataFrame, df2: pd.DataFrame, label: str,
                  name1: str, name2: str, horizon: Optional[int] = None):
    a = df1.copy()
    b = df2.copy()
    if horizon is not None:
        a = a[a["horizon_w"] == horizon]
        b = b[b["horizon_w"] == horizon]

    # Hit-rate test
    hits1, n1 = int(a["win"].sum()), int(len(a))
    hits2, n2 = int(b["win"].sum()), int(len(b))
    diff_hr, z_hr, p_hr = hitrate_test(hits1, n1, hits2, n2)

    # Sharpe test: align per-trade PnL by time order; trim to min length
    m = min(len(a), len(b))
    r1 = a["pnl_units"].to_numpy()[:m]
    r2 = b["pnl_units"].to_numpy()[:m]
    sr = sharpe_jk_memmel(r1, r2)

    print(f"\n===== {label} — {name1} vs {name2} =====")
    print(f"Counts: {name1} n={n1}, {name2} n={n2}")
    # Hit rate
    hr1 = hits1 / n1 if n1 else np.nan
    hr2 = hits2 / n2 if n2 else np.nan
    print(f"Hit rate: {name1}={hr1:.4f}, {name2}={hr2:.4f}, diff={diff_hr:.4f}, z={z_hr:.3f}, p={p_hr:.4g}")
    print(" → " + label_sig(name1, name2, diff_hr, p_hr))
    # Sharpe
    print(f"Sharpe:  {name1}={sr['SR1']:.6f}, {name2}={sr['SR2']:.6f}, diff={sr['diff']:.6f}, z={sr['z']:.3f}, p={sr['p']:.4g}")
    print(" → " + label_sig(name1, name2, sr['diff'], sr['p']))

def compare_pair(name1: str, path1: str, name2: str, path2: str):
    df1 = load_trades(path1)
    df2 = load_trades(path2)

    # OVERALL
    analyze_block(df1, df2, label="OVERALL (flip-exit, q=10%)", name1=name1, name2=name2, horizon=None)
    # PER HORIZON
    for H in (1, 4, 12):
        analyze_block(df1, df2, label=f"H={H} (flip-exit, q=10%)", name1=name1, name2=name2, horizon=H)

def main():
    # HMM_LSTM vs AR(p)
    compare_pair("HMM-LSTM", HMM_ALL, "AR(p)", ARP_ALL)
    # HMM_LSTM vs AR-X
    compare_pair("HMM-LSTM", HMM_ALL, "AR-X", ARX_ALL)

if __name__ == "__main__":
    main()


# In[ ]:


import os
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------- Config --------------------
TRADE_FOLDER = "/Users/anwarouni/Downloads/Thesis/Output"
MODELS = ["ARP", "ARX", "LSTM", "HMM_LSTM"]
TOPQ = 0.10

# Weekly grid + event matching
WEEK_FREQ = "W-FRI"             # pick a consistent week anchor; change to W-MON if you prefer
MATCH_WINDOW_WEEKS = 1          # entries/exits within ±1 week are considered "aligned"

# Output
OUTDIR = Path("/Users/anwarouni/Downloads/Thesis/Output/TradeAlignment")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -------------------- Helpers --------------------
def load_trades(model: str) -> pd.DataFrame:
    model_dir = Path(TRADE_FOLDER) / f"Trading Backtest ({model} q{int(TOPQ*100)} flip)"
    f = model_dir / f"ALL_trades_q{int(TOPQ*100)}_flip.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing: {f}")
    df = pd.read_csv(f, parse_dates=["entry_date", "exit_date"]).sort_values("entry_date").reset_index(drop=True)
    # defensive: ensure exit after entry
    df = df[df["exit_date"] >= df["entry_date"]].copy()
    return df

def weekly_inmarket_series(df: pd.DataFrame, week_index: pd.DatetimeIndex) -> pd.Series:
    """Return boolean Series over week_index: True if in market that week."""
    s = pd.Series(False, index=week_index)
    if df.empty:
        return s
    # mark weeks covered by each trade
    for _, r in df.iterrows():
        # cover inclusive bounds on this weekly grid
        rng = pd.date_range(r["entry_date"], r["exit_date"], freq=WEEK_FREQ)
        s.loc[s.index.isin(rng)] = True
    return s

def segment_lengths(mask: pd.Series, value: bool) -> list[int]:
    """Lengths (in weeks) of consecutive segments where mask == value."""
    if mask.empty:
        return []
    runs = []
    run = 0
    for v in mask.values:
        if v == value:
            run += 1
        else:
            if run > 0:
                runs.append(run)
            run = 0
    if run > 0:
        runs.append(run)
    return runs

def match_events(a_dates: pd.Series, b_dates: pd.Series, window_weeks: int) -> pd.DataFrame:
    """
    Greedy nearest-neighbour matching of dates in a_dates to b_dates within ±window_weeks.
    Returns a DataFrame with columns: a_date, b_date (or NaT), gap_weeks (signed, b - a).
    """
    a_dates = pd.to_datetime(a_dates).sort_values().reset_index(drop=True)
    b_dates = pd.to_datetime(b_dates).sort_values().reset_index(drop=True)
    used = np.zeros(len(b_dates), dtype=bool)

    rows = []
    for a in a_dates:
        # find nearest b (by absolute week diff) that is unused and within window
        if b_dates.empty:
            rows.append({"a_date": a, "b_date": pd.NaT, "gap_weeks": np.nan})
            continue
        diffs = (b_dates - a).dt.days / 7.0
        diffs_abs = diffs.abs()
        # set used to inf so they won't be picked
        diffs_abs.values[used] = np.inf
        j = diffs_abs.values.argmin()
        best_gap = diffs.values[j]
        if np.isfinite(diffs_abs.values[j]) and abs(best_gap) <= window_weeks:
            used[j] = True
            rows.append({"a_date": a, "b_date": b_dates.iloc[j], "gap_weeks": best_gap})
        else:
            rows.append({"a_date": a, "b_date": pd.NaT, "gap_weeks": np.nan})
    return pd.DataFrame(rows)

def nearest_gap_for_unmatched(unmatched_dates: pd.Series, other_dates: pd.Series) -> pd.Series:
    """
    For each unmatched date, compute signed gap (in weeks) to nearest date in other_dates.
    Positive means other is after, negative means before.
    """
    unmatched_dates = pd.to_datetime(unmatched_dates).sort_values().reset_index(drop=True)
    other_dates = pd.to_datetime(other_dates).sort_values().reset_index(drop=True)
    if unmatched_dates.empty or other_dates.empty:
        return pd.Series(dtype=float)
    gaps = []
    for d in unmatched_dates:
        diffs = (other_dates - d).dt.days / 7.0
        j = diffs.abs().values.argmin()
        gaps.append(diffs.iloc[j])
    return pd.Series(gaps, name="nearest_gap_weeks")

# -------------------- Main --------------------
def main():
    # Load trades
    trades = {m: load_trades(m) for m in MODELS}
    base = "HMM_LSTM"
    assert base in trades, "HMM_LSTM trades are required as the reference."

    # Build a global weekly index covering all models
    min_date = min(df["entry_date"].min() for df in trades.values() if not df.empty)
    max_date = max(df["exit_date"].max()  for df in trades.values() if not df.empty)
    week_index = pd.date_range(min_date, max_date, freq=WEEK_FREQ)

    # Weekly in-market series per model
    weekly = {m: weekly_inmarket_series(trades[m], week_index) for m in MODELS}

    # ---- Weekly overlap summaries vs HMM_LSTM ----
    weekly_rows = []
    base_s = weekly[base]
    for m in MODELS:
        s = weekly[m]
        both = base_s & s
        either = base_s | s
        jaccard = (both.sum() / max(1, either.sum())) * 100.0
        cond_on_base = (both.sum() / max(1, base_s.sum())) * 100.0

        non_overlap = either & (~both)            # weeks where at least one is in-market but not both
        both_runs  = segment_lengths(both, True)
        non_runs   = segment_lengths(non_overlap, True)

        weekly_rows.append({
            "model": m,
            "weeks_in_base": int(base_s.sum()),
            "weeks_in_model": int(s.sum()),
            "weeks_both": int(both.sum()),
            "weeks_either": int(either.sum()),
            "jaccard_overlap_pct": jaccard,
            "conditional_overlap_pct": cond_on_base,
            "median_overlap_run_weeks": int(np.median(both_runs)) if both_runs else 0,
            "median_nonoverlap_run_weeks": int(np.median(non_runs)) if non_runs else 0,
        })

    weekly_df = pd.DataFrame(weekly_rows).sort_values("model")
    weekly_df.to_csv(OUTDIR / "weekly_overlap_vs_HMM_LSTM.csv", index=False)

    # ---- Event-level alignment (entries & exits), vs HMM_LSTM ----
    def summarize_event_alignment(event: str):
        rows = []
        base_dates = trades[base][f"{event}_date"]

        for m in MODELS:
            other_dates = trades[m][f"{event}_date"]

            match_df = match_events(base_dates, other_dates, MATCH_WINDOW_WEEKS)
            matched = match_df["b_date"].notna()
            pct_aligned = 100.0 * matched.mean()

            # gaps for matched pairs
            gaps_matched = match_df.loc[matched, "gap_weeks"].astype(float)
            # nearest gaps for unmatched (how far to the closest event in the other model)
            gaps_unmatched = nearest_gap_for_unmatched(
                match_df.loc[~matched, "a_date"], other_dates
            )

            # summary row
            rows.append({
                "model": m,
                f"n_{event}_base": len(base_dates),
                f"n_{event}_model": len(other_dates),
                f"{event}_aligned_pct": pct_aligned,
                f"{event}_matched_gap_median_w": np.median(gaps_matched) if not gaps_matched.empty else np.nan,
                f"{event}_matched_gap_mean_w": gaps_matched.mean() if not gaps_matched.empty else np.nan,
                f"{event}_unmatched_count": int((~matched).sum()),
                f"{event}_unmatched_nearest_gap_median_w": np.median(gaps_unmatched) if not gaps_unmatched.empty else np.nan,
                f"{event}_unmatched_nearest_gap_mean_w": gaps_unmatched.mean() if not gaps_unmatched.empty else np.nan,
            })

            # dump detailed matches to CSV per model/event
            det = match_df.copy()
            det.columns = [f"{event}_base", f"{event}_model", f"{event}_gap_weeks"]
            det.to_csv(OUTDIR / f"detail_{event}_matches_{m}_vs_{base}.csv", index=False)

        return pd.DataFrame(rows).sort_values("model")

    entry_df = summarize_event_alignment("entry")
    exit_df  = summarize_event_alignment("exit")

    entry_df.to_csv(OUTDIR / "event_alignment_entry_vs_HMM_LSTM.csv", index=False)
    exit_df.to_csv(OUTDIR / "event_alignment_exit_vs_HMM_LSTM.csv", index=False)

    # ---- Print concise console views ----
    print("\n=== Weekly Overlap vs HMM_LSTM (grid: {}, window={}w) ===".format(WEEK_FREQ, MATCH_WINDOW_WEEKS))
    print(weekly_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n=== Entry Alignment vs HMM_LSTM (±{}w) ===".format(MATCH_WINDOW_WEEKS))
    print(entry_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n=== Exit Alignment vs HMM_LSTM (±{}w) ===".format(MATCH_WINDOW_WEEKS))
    print(exit_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

if __name__ == "__main__":
    main()

