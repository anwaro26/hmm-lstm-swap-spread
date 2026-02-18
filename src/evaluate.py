#!/usr/bin/env python
# coding: utf-8

# Diebold Mariano

# In[ ]:


import re, math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List


# ------------------------ Paths -------------------------
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

HMM_ROOT = Path(os.getenv("HMM_ROOT", REPO_ROOT / "Output"))
NEW_LSTM_ROOT = Path(os.getenv("LSTM_ROOT", HMM_ROOT / "LSTM predictionsTEST"))
OUTDIR = Path(os.getenv("DM_OUTDIR", REPO_ROOT / "DM_tests_HMM_vs_LSTM_NEW"))
OUTDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- Config ------------------------------------------
SPLITS    = [60, 70, 80, 90]
HORIZONS  = [1, 4, 12]

OVERLAPPING = True   # q = H-1
USE_HLN     = True   # HLN correction ON

# ----------------------------- Discovery ---------------------------------------
def discover_csvs(root: Path, include_tokens: List[str], exclude_tokens: List[str]) -> List[Path]:
    hits = []
    for p in root.rglob("*.csv"):
        name = p.name.lower()
        if all(tok in name for tok in include_tokens) and not any(tok in name for tok in exclude_tokens):
            hits.append(p)
    return sorted(hits)

def _int_after(token: str, s: str) -> Optional[int]:
    i = s.lower().rfind(token.lower())
    if i < 0: return None
    tail = s[i+len(token):]
    m = re.search(r"(\d+)", tail)
    return int(m.group(1)) if m else None

def parse_split_h(filename: str) -> Tuple[Optional[int], Optional[int]]:
    split = _int_after("split", filename)
    h = _int_after("_h", filename) or _int_after("h", filename)
    return split, h

def index_files(paths: List[Path]) -> Dict[Tuple[int,int], Path]:
    out: Dict[Tuple[int,int], Path] = {}
    for p in paths:
        s, h = parse_split_h(p.name)
        if s in SPLITS and h in HORIZONS:
            out[(s, h)] = p
    return out

# HMM-LSTM files (e.g., HMM_LSTM_split_60_H1.csv)
HMM_FILES = discover_csvs(HMM_ROOT, include_tokens=["hmm","lstm","split"], exclude_tokens=["dm","test"])
HMM_MAP   = index_files(HMM_FILES)

# Plain LSTM files in the NEW folder (e.g., LSTM_split_60_H1.csv)
# NOTE: 
LSTM_FILES = discover_csvs(NEW_LSTM_ROOT, include_tokens=["lstm","split"], exclude_tokens=["hmm"])
LSTM_MAP   = index_files(LSTM_FILES)

print("Matched HMM-LSTM:", {k: v.name for k, v in HMM_MAP.items()})
print("Matched LSTM (NEW):", {k: v.name for k, v in LSTM_MAP.items()})

# --------------------------- Column helpers -------------------------------------
ACTUAL_CANDS       = ["y_true", "actual", "y", "target", "swap spread", "truth", "true"]
FORECAST_ANY       = ["y_hat", "forecast", "yhat", "pred", "prediction"]
FORECAST_HMM_PREF  = ["forecast hmm lstm", "hmm_lstm_forecast", "y_hat_hmm", "hmm yhat", "hmm"]

def pick_actual_col(df: pd.DataFrame) -> str:
    cols = [str(c).strip() for c in df.columns if not str(c).lower().startswith("unnamed")]
    for cand in ACTUAL_CANDS:
        for c in cols:
            if c.lower() == cand:
                return c
    for c in cols:  # fallback: first numeric
        if pd.api.types.is_numeric_dtype(df[c]): return c
    raise ValueError("Could not identify Actual column.")

def pick_forecast_col(df: pd.DataFrame, model: str) -> str:
    cols  = [str(c).strip() for c in df.columns if not str(c).lower().startswith("unnamed")]
    lower = [c.lower() for c in cols]
    if model.lower() == "hmm_lstm":
        for pref in FORECAST_HMM_PREF:
            for i, c in enumerate(lower):
                if pref in c: return cols[i]
        for i, c in enumerate(lower):
            if any(k in c for k in FORECAST_ANY): return cols[i]
    else:  # plain LSTM
        for i, c in enumerate(lower):
            if c == "y_hat": return cols[i]
        for i, c in enumerate(lower):
            if any(k in c for k in FORECAST_ANY) and ("hmm" not in c): return cols[i]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if numeric: return numeric[-1]
    raise ValueError(f"Could not identify Forecast column for model={model}.")

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

    a_col = pick_actual_col(df)
    f_col = pick_forecast_col(df, model)
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
    """
    One-sided ALT='greater': tests HMM-LSTM has lower MSE than LSTM.
    d_t = e_LSTM^2 - e_HMM^2 ; DM = dbar / se_NW.
    """
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

    if use_hln:
        # Harvey–Leybourne–Newbold correction
        hln = math.sqrt((T + 1 - 2*h + (h*(h-1))/T) / T) if T > 0 else np.nan
        dm *= hln

    from math import erf, sqrt
    def norm_cdf(x): return 0.5*(1 + erf(x / sqrt(2)))
    p = 1 - norm_cdf(dm)  # one-sided 'greater'
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
by_split_path = OUTDIR / "DM_HMM_LSTM_vs_LSTM_NEW_by_split_MSE_one_sided_HLN.csv"
by_split.to_csv(by_split_path, index=False)
print(f"Saved per-split results → {by_split_path}")

# --------------------------- Pooled per horizon ---------------------------------
pooled_rows = []
for h in HORIZONS:
    eL_all, eH_all = [], []
    for s in SPLITS:
        p_l = LSTM_MAP.get((s, h)); p_h = HMM_MAP.get((s, h))
        if p_l is None or p_h is None: continue
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
pooled_path = OUTDIR / "DM_HMM_LSTM_vs_LSTM_NEW_by_horizon_MSE_one_sided_HLN.csv"
pooled.to_csv(pooled_path, index=False)
print(f"Saved pooled results → {pooled_path}")

# --------------------------- Console preview ------------------------------------
def fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return "—"
    return f"{x:.4f}" if isinstance(x, float) else str(x)

print("\n============= ONE-SIDED DM (HMM-LSTM better) — MSE (HLN ON) =============")
for h in HORIZONS:
    sub = by_split[by_split["H"] == h]
    if sub.empty: continue
    print(f"\nH = {h}")
    print("Split  N    MSE_LSTM  MSE_HMM  dbar     DM      p     sig  q")
    for _, r in sub.iterrows():
        print(f"{fmt(r['Split']):>4}  {fmt(r['N']):>4}  {fmt(r['MSE_LSTM']):>9}  {fmt(r['MSE_HMM_LSTM']):>8}  "
              f"{fmt(r['dbar']):>7}  {fmt(r['DM']):>6}  {fmt(r['p_value']):>6}  {r['sig']:>3}  {fmt(r['q_lag']):>1}")

print("\n================ Pooled per horizon (one-sided) ================")
for _, r in pooled.iterrows():
    print(f"H={fmt(r['H'])} | N={fmt(r.get('N_total',0))} | "
          f"MSE: LSTM={fmt(r.get('MSE_LSTM'))}, HMM={fmt(r.get('MSE_HMM_LSTM'))} | "
          f"dbar={fmt(r.get('dbar'))} | DM={fmt(r.get('DM'))} (p={fmt(r.get('p_value'))}) {r.get('sig','')} "
          f"| q={fmt(r['q_lag'])} | HLN={r['HLN']}")


# In[ ]:


import re, math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List

# ----------------------------- Config / Paths -----------------------------------
ROOT      = Path("/Users/anwarouni/Downloads/Thesis/Output")
OUTDIR    = ROOT / "DM tests"
OUTDIR.mkdir(parents=True, exist_ok=True)

SPLITS    = [60, 70, 80, 90]
HORIZONS  = [1, 4, 12]

# Candidate vs Baseline
CAND_NAME      = "HMM_LSTM"
BASE_NAME      = "ARMA11"   # plain ARMA(1,1), not ARMAX, not "arma11x"

OVERLAPPING   = True        # q = H-1 if True; else q=0
USE_HLN       = True        # HLN correction ON
ALT           = "greater"   # one-sided: dbar > 0 ⇒ HMM-LSTM better (lower MSE)


def discover_csvs_regex(root: Path,
                        include_patterns: List[str],
                        exclude_patterns: List[str]) -> List[Path]:
    """
    Regex-based discovery. A file is selected iff it matches ALL include_patterns
    and NONE of the exclude_patterns. Matching is case-insensitive.
    """
    hits = []
    for p in root.rglob("*.csv"):
        name = p.name.lower()
        ok_inc = all(re.search(pat, name, flags=re.IGNORECASE) for pat in include_patterns)
        ok_exc = any(re.search(pat, name, flags=re.IGNORECASE) for pat in exclude_patterns)
        if ok_inc and not ok_exc:
            hits.append(p)
    return sorted(hits)

def _int_after(token: str, s: str) -> Optional[int]:
    i = s.lower().rfind(token.lower())
    if i < 0: return None
    tail = s[i+len(token):]
    m = re.search(r"(\d+)", tail)
    return int(m.group(1)) if m else None

def parse_split_h(filename: str) -> Tuple[Optional[int], Optional[int]]:
    split = _int_after("split", filename)
    h = _int_after("_h", filename) or _int_after("h", filename)
    return split, h

def index_files(paths: List[Path]) -> Dict[Tuple[int,int], Path]:
    out: Dict[Tuple[int,int], Path] = {}
    for p in paths:
        s, h = parse_split_h(p.name)
        if s in SPLITS and h in HORIZONS:
            out[(s, h)] = p
    return out

# HMM-LSTM files (e.g., HMM_LSTM_split_60_H1.csv)
HMM_FILES = discover_csvs_regex(
    ROOT,
    include_patterns=[r"hmm", r"lstm", r"split"],
    exclude_patterns=[r"dm", r"test"]
)
HMM_MAP   = index_files(HMM_FILES)


ARMA11_FILES = discover_csvs_regex(
    ROOT,
    include_patterns=[r"(^|[_-])arma11([_-])", r"split"],
    exclude_patterns=[r"armax", r"arma11x", r"hmm", r"lstm", r"arx", r"garch"]
)
ARIMA11_FILES = discover_csvs_regex(
    ROOT,
    include_patterns=[r"(^|[_-])arima11([_-])", r"split"],
    exclude_patterns=[r"armax", r"arma11x", r"hmm", r"lstm", r"arx", r"garch"]
)
BASE_FILES = sorted(set(ARMA11_FILES + ARIMA11_FILES), key=lambda p: p.as_posix())
BASE_MAP   = index_files(BASE_FILES)

print("Matched HMM-LSTM:", {k: v.name for k, v in HMM_MAP.items()})
print("Matched ARMA(1,1):", {k: v.name for k, v in BASE_MAP.items()})

# ----------------------------- Column helpers -----------------------------------
ACTUAL_CANDS = ["y_true", "actual", "y", "target", "swap spread", "truth", "true"]
FORECAST_ANY = ["y_hat", "forecast", "yhat", "pred", "prediction"]

FORECAST_PREFS = {
    "HMM_LSTM": ["forecast hmm lstm", "hmm_lstm_forecast", "y_hat_hmm", "yhat_hmm", "hmm yhat", "hmm"],
    "ARMA11":   ["arma11", "arima11", "arma(1,1)", "arima(1,1)", "y_hat_arma11", "yhat_arma11",
                 "forecast_arma11", "y_hat_arma"],
}

def pick_actual_col(df: pd.DataFrame) -> str:
    cols = [str(c).strip() for c in df.columns if not str(c).lower().startswith("unnamed")]
    for cand in ACTUAL_CANDS:
        for c in cols:
            if c.lower() == cand:
                return c
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]): return c
    raise ValueError("Could not identify the Actual column (e.g., y_true).")

def pick_forecast_col(df: pd.DataFrame, model_key: str) -> str:
    cols  = [str(c).strip() for c in df.columns if not str(c).lower().startswith("unnamed")]
    lower = [c.lower() for c in cols]
    # Prefer model-specific tokens
    for pref in FORECAST_PREFS.get(model_key, []):
        for i, c in enumerate(lower):
            if pref in c:
                return cols[i]
    # Fallback: generic forecast column (safe if each CSV is model-specific)
    for i, c in enumerate(lower):
        if any(k in c for k in FORECAST_ANY):
            if model_key != "HMM_LSTM" and "hmm" in c:
                continue
            return cols[i]
    # Last resort: last numeric column
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if numeric: return numeric[-1]
    raise ValueError(f"Could not identify Forecast column for model={model_key}.")

def load_series(path: Path, model_key: str) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    # Optional Date index
    date_col = None
    for c in df.columns:
        if str(c).lower() in ["date", "time", "timestamp"]:
            date_col = c; break
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)

    a_col = pick_actual_col(df)
    f_col = pick_forecast_col(df, model_key)

    # DEBUG (uncomment if needed):
    # print(f"[{model_key}] file={path.name} | actual={a_col} | forecast={f_col}")

    y    = pd.to_numeric(df[a_col], errors="coerce")
    yhat = pd.to_numeric(df[f_col], errors="coerce")
    s = pd.concat([y.rename("y"), yhat.rename("yhat")], axis=1).dropna()
    return s["y"], s["yhat"]

# ----------------------------- Newey–West variance -------------------------------
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

# ----------------------------- DM over MSE (HLN ON) ------------------------------
def dm_test_mse(e_base: np.ndarray, e_cand: np.ndarray, h: int,
                overlapping: bool = OVERLAPPING, use_hln: bool = USE_HLN):
    """
    Returns: DM_stat, p_value (one-sided ALT='greater'), T, dbar, se_NW, q_lag
    d_t = e_base^2 - e_cand^2 ; ALT 'greater' tests Candidate (HMM-LSTM) better (lower MSE).
    """
    assert len(e_base) == len(e_cand)
    T = len(e_base)
    d = (e_base**2) - (e_cand**2)
    q = max(h - 1, 0) if overlapping else 0

    var = nw_variance(d, q)
    if not np.isfinite(var) or var <= 0:
        return np.nan, np.nan, T, float(np.mean(d)), np.nan, q

    dbar = float(np.mean(d))
    se   = math.sqrt(var)
    dm   = dbar / se

    if use_hln:
        # Harvey–Leybourne–Newbold small-sample correction
        hln = math.sqrt((T + 1 - 2*h + (h*(h-1))/T) / T) if T > 0 else np.nan
        dm *= hln

    from math import erf, sqrt
    def norm_cdf(x): return 0.5*(1 + erf(x / sqrt(2)))
    p = 1 - norm_cdf(dm)  # one-sided: greater
    return dm, p, T, dbar, se, q

def stars(p):
    return "***" if (p is not None and p < 0.01) else ("**" if (p is not None and p < 0.05) else ("*" if (p is not None and p < 0.10) else ""))

def nan_row(extra: dict) -> dict:
    base = {
        "N": 0,
        f"MSE_{BASE_NAME}": np.nan, f"MSE_{CAND_NAME}": np.nan,
        f"RMSE_{BASE_NAME}": np.nan, f"RMSE_{CAND_NAME}": np.nan,
        "dbar": np.nan, "SE_NW": np.nan, "q_lag": np.nan,
        "DM": np.nan, "p_value": np.nan, "sig": "",
        "ALT": ALT, "HLN": USE_HLN, "overlapping": OVERLAPPING
    }
    base.update(extra); return base

# ----------------------------- Per split × horizon ------------------------------
rows = []
for s in SPLITS:
    for h in HORIZONS:
        p_b = BASE_MAP.get((s, h)); p_c = HMM_MAP.get((s, h))
        if p_b is None or p_c is None:
            rows.append(nan_row({"Split": s, "H": h, "note": "missing file(s)"}))
            continue
        try:
            y_b, f_b = load_series(p_b, model_key=BASE_NAME)
            y_c, f_c = load_series(p_c, model_key=CAND_NAME)
            df = pd.concat([y_b.rename("y"), f_b.rename("f_base"),
                            y_c.rename("y2"), f_c.rename("f_cand")], axis=1).dropna()
            if df.empty:
                rows.append(nan_row({"Split": s, "H": h, "note": "no overlap after merge"}))
                continue

            y   = df["y"]  # equals y2 after merge
            eB  = (y - df["f_base"]).to_numpy()
            eC  = (y - df["f_cand"]).to_numpy()

            DM, p, T, dbar, se, q = dm_test_mse(eB, eC, h=h)

            mse_b = float(np.mean(eB**2)); mse_c = float(np.mean(eC**2))
            rmse_b = float(np.sqrt(mse_b)); rmse_c = float(np.sqrt(mse_c))

            rows.append({
                "Split": s, "H": h, "N": int(T),
                f"MSE_{BASE_NAME}": mse_b, f"MSE_{CAND_NAME}": mse_c,
                f"RMSE_{BASE_NAME}": rmse_b, f"RMSE_{CAND_NAME}": rmse_c,
                "dbar": dbar, "SE_NW": se, "q_lag": q,
                "DM": DM, "p_value": p, "sig": stars(p),
                "ALT": ALT, "HLN": USE_HLN, "overlapping": OVERLAPPING
            })
        except Exception as ex:
            rows.append(nan_row({"Split": s, "H": h, "note": "exception", "error": str(ex)}))

by_split = pd.DataFrame(rows).sort_values(["H","Split"]).reset_index(drop=True)
by_split_path = OUTDIR / f"DM_{CAND_NAME}_vs_{BASE_NAME}_by_split_MSE_one_sided_HLN.csv"
by_split.to_csv(by_split_path, index=False)
print(f"Saved per-split results → {by_split_path}")

# ----------------------------- Pooled per horizon --------------------------------
pooled_rows = []
for h in HORIZONS:
    eB_all, eC_all = [], []
    for s in SPLITS:
        p_b = BASE_MAP.get((s, h)); p_c = HMM_MAP.get((s, h))
        if p_b is None or p_c is None: continue
        y_b, f_b = load_series(p_b, model_key=BASE_NAME)
        y_c, f_c = load_series(p_c, model_key=CAND_NAME)
        df = pd.concat([y_b.rename("y"), f_b.rename("f_base"),
                        y_c.rename("y2"), f_c.rename("f_cand")], axis=1).dropna()
        if not df.empty:
            y   = df["y"]
            eB  = (y - df["f_base"]).to_numpy()
            eC  = (y - df["f_cand"]).to_numpy()
            eB_all.append(eB); eC_all.append(eC)

    if not eB_all:
        pooled_rows.append(nan_row({"H": h, "N_total": 0, "note": "no data"}))
        continue

    eB = np.concatenate(eB_all, axis=0)
    eC = np.concatenate(eC_all, axis=0)

    DM, p, T, dbar, se, q = dm_test_mse(eB, eC, h=h)

    mse_b = float(np.mean(eB**2)); mse_c = float(np.mean(eC**2))
    rmse_b = float(np.sqrt(mse_b)); rmse_c = float(np.sqrt(mse_c))

    pooled_rows.append({
        "H": h, "N_total": int(T),
        f"MSE_{BASE_NAME}": mse_b, f"MSE_{CAND_NAME}": mse_c,
        f"RMSE_{BASE_NAME}": rmse_b, f"RMSE_{CAND_NAME}": rmse_c,
        "dbar": dbar, "SE_NW": se, "q_lag": q,
        "DM": DM, "p_value": p, "sig": stars(p),
        "ALT": ALT, "HLN": USE_HLN, "overlapping": OVERLAPPING
    })

pooled = pd.DataFrame(pooled_rows).sort_values("H").reset_index(drop=True)
pooled_path = OUTDIR / f"DM_{CAND_NAME}_vs_{BASE_NAME}_by_horizon_MSE_one_sided_HLN.csv"
pooled.to_csv(pooled_path, index=False)
print(f"Saved pooled results → {pooled_path}")

# ----------------------------- Console preview -----------------------------------
def fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return "—"
    return f"{x:.4f}" if isinstance(x, float) else str(x)

print(f"\n============= ONE-SIDED DM ({CAND_NAME} < {BASE_NAME}) — MSE (HLN ON) =============")
for h in HORIZONS:
    sub = by_split[by_split["H"] == h]
    if sub.empty: continue
    print(f"\nH = {h}")
    print(f"Split  N    MSE_{BASE_NAME:>8}  MSE_{CAND_NAME:>12}  dbar     DM      p     sig  q")
    for _, r in sub.iterrows():
        print(f"{int(r['Split']):>4}  {int(r['N']):>4}  {fmt(r[f'MSE_{BASE_NAME}']):>12}  {fmt(r[f'MSE_{CAND_NAME}']):>12}  "
              f"{fmt(r['dbar']):>7}  {fmt(r['DM']):>6}  {fmt(r['p_value']):>6}  {r['sig']:>3}  {fmt(r['q_lag']):>1}")

print("\n================ Pooled per horizon (one-sided) ================")
for _, r in pooled.iterrows():
    print(f"H={int(r['H'])} | N={int(r.get('N_total',0))} | "
          f"MSE: ARMA(1,1)={fmt(r.get(f'MSE_{BASE_NAME}'))}, HMM-LSTM={fmt(r.get(f'MSE_{CAND_NAME}'))} | "
          f"dbar={fmt(r.get('dbar'))} | DM={fmt(r.get('DM'))} (p={fmt(r.get('p_value'))}) {r.get('sig','')} "
          f"| q={fmt(r['q_lag'])} | HLN={r['HLN']}")

