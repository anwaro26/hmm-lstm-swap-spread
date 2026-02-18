#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from scipy import stats

import matplotlib.pyplot as plt
from src.paths import DATA_SAMPLE, RESULTS_DIR

# Read the Excel file
df = pd.read_excel('/Users/anwarouni/Downloads/Thesis/Data/Data_final.xlsx')

# Display first few rows to verify import
df.head()


# In[14]:


def check_stationarity(data):
    # Create a dictionary to store results
    results = {}

    # Get numerical columns only (exclude Date)
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns

    for column in numerical_cols:
        # Perform ADF test
        adf_test = adfuller(data[column].dropna())

        # Store results
        results[column] = {
            'ADF Statistic': adf_test[0],
            'p-value': adf_test[1],
            'Critical values': adf_test[4]
        }

    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict({k: v['p-value'] for k, v in results.items()}, 
                                      orient='index', 
                                      columns=['p-value'])

    # Add stationarity conclusion
    results_df['Is Stationary'] = results_df['p-value'] < 0.05

    # Sort by p-value
    results_df = results_df.sort_values('p-value')

    return results_df

# Run the analysis
stationarity_results = check_stationarity(df)
print("Stationarity Analysis Results:")
print(stationarity_results)


# In[15]:


def analyze_stationarity(data, column):
    # Setup the plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Plot 1: Time series plot
    axes[0].plot(data['Date'], data[column])
    axes[0].set_title(f'Time Series Plot: {column}')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Value')

    # Plot 2: Rolling statistics
    rolling_mean = data[column].rolling(window=52).mean()
    rolling_std = data[column].rolling(window=52).std()

    axes[1].plot(data['Date'], data[column], label='Original')
    axes[1].plot(data['Date'], rolling_mean, label='Rolling Mean')
    axes[1].plot(data['Date'], rolling_std, label='Rolling Std')
    axes[1].set_title(f'Rolling Statistics: {column}')
    axes[1].legend()

    plt.tight_layout()

    # Run ADF tests with different specifications
    specifications = {
        'No Constant': 'n',
        'Constant': 'c',
        'Constant + Trend': 'ct'
    }

    print(f"\nADF Test Results for {column}:")
    print("=" * 50)

    results = {}
    conclusion = ""
    best_spec = None
    min_aic = float('inf')

    for name, regression in specifications.items():
        adf_result = adfuller(data[column].dropna(), regression=regression)
        # Calculate AIC
        nobs = len(data[column].dropna())
        k = 1 + (regression != 'n') + (regression == 'ct')  # parameters: intercept and trend
        aic = np.log(adf_result[1]) * nobs + 2 * (k + adf_result[2])  # AIC calculation

        results[name] = {
            'ADF Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Critical values': adf_result[4],
            'Used Lags': adf_result[2],
            'AIC': aic
        }

        # Track best specification based on AIC
        if aic < min_aic:
            min_aic = aic
            best_spec = name

        print(f"\n{name}:")
        print(f"ADF Statistic: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        print(f"AIC: {aic:.4f}")
        print("Critical values:")
        for key, value in adf_result[4].items():
            print(f"\t{key}: {value:.4f}")

        # Determine stationarity type based on best AIC model
        if adf_result[1] < 0.05:
            if name == best_spec:
                if name == 'Constant + Trend':
                    conclusion = "Trend Stationary"
                elif name == 'Constant':
                    conclusion = "Stationary with Constant"
                else:
                    conclusion = "Stationary"

    if conclusion == "":
        conclusion = "Not Stationary"

    print(f"\nBest specification (lowest AIC): {best_spec}")
    print(f"Conclusion: {conclusion}")

    return results, conclusion

# Example usage for a few key variables
variables_to_check = ['VSTOXX', 'Swap Spread', 'Credit_Risk', 'ECBrate', 
                     'German_CPI YoY', 'Euribor-OIS', 'Yield_Slope',
                     'German_UnemploymentRate']
conclusions = {}
for var in variables_to_check:
    _, conclusions[var] = analyze_stationarity(df, var)
    plt.show()

print("\nSummary of Conclusions:")
print("=" * 50)
for var, conclusion in conclusions.items():
    print(f"{var}: {conclusion}")



# In[16]:


# Calculate correlation matrix with the variable list
corr_matrix = df[variables_to_check].corr()

# Create a heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, 
            annot=True,           # Show correlation values
            cmap='coolwarm',      # Color scheme
            center=0,             # Center the colormap at 0
            fmt='.2f',           # Format correlation values to 2 decimal places
            square=True)         # Make the plot square-shaped

plt.title('Correlation Matrix of Variables', pad=20)
plt.tight_layout()
plt.show()


# In[17]:


# Filter correlations for Swap Spread
swap_spread_corr = corr_matrix['Swap Spread'].abs()
significant_corr = swap_spread_corr[swap_spread_corr > 0.1]
significant_corr = significant_corr.sort_values(ascending=False)

print("Variables correlated with Swap Spread (|correlation| > 0.1):")
for var, corr in significant_corr.items():
    if var != 'Swap Spread':  # Exclude self-correlation
        print(f"{var}: {corr_matrix['Swap Spread'][var]:.3f}")


# In[19]:


# Identify non-stationary variables
non_stationary_vars = stationarity_results[~stationarity_results['Is Stationary']].index

# Take the first difference for non-stationary variables
df_diff = df.copy()
for var in non_stationary_vars:
    df_diff[var] = df_diff[var].diff()

# Drop the first row due to NaN values after differencing
df_diff = df_diff.dropna()

# Rerun stationarity tests for the differenced variables
stationarity_results_diff = check_stationarity(df_diff[non_stationary_vars])

print("Stationarity Analysis Results After Differencing:")
print(stationarity_results_diff)


# In[20]:


# Remove the first row from the dataframe
df_aligned = df.iloc[1:].reset_index(drop=True)

# Show the number of values per column
print("Number of values per column:")
print(df_aligned.count())

# Display the first 5 rows of the new dataframe
print("\nFirst 5 rows of the new dataframe:")
print(df_aligned.head())


# In[21]:


# Write the dataframe to an Excel file
output_file = 'output_data.xlsx'
df.to_excel(output_file, index=False)

# Read back the Excel file to verify and summarize
df_summary = pd.read_excel(output_file)

# Display the first 5 rows and the columns
print(f"Data has been written to {output_file}")
print("\nSummary of the first 5 rows:")
print(df_summary.head())
print("\nColumns in the Excel file:")
print(df_summary.columns.tolist())


# Plots

# In[3]:


# --- Swap spread plots: (1) level, (2) first difference — on separate figures ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
PATH_TO_EXCEL = "/Users/anwarouni/Downloads/Thesis/Code/output_data.xlsx"
DATE_COL      = "Date"
TARGET_COL    = "Swap Spread"

# If your series isn’t strictly weekly but you want weekly spacing, set True:
RESAMPLE_TO_WEEKLY = False
RESAMPLE_RULE      = "W-FRI"    # weekly (Friday)
# ==================================================

# ---- Load & prep ----
df = pd.read_excel(PATH_TO_EXCEL)

if DATE_COL in df.columns:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).set_index(DATE_COL)

y = df[TARGET_COL].astype(float).dropna()

if RESAMPLE_TO_WEEKLY:
    # keep last obs within each week; forward-fill if needed
    y = y.resample(RESAMPLE_RULE).last().ffill()

dy = y.diff().dropna()
sample_str = f"{y.index.min():%Y-%m-%d} to {y.index.max():%Y-%m-%d}"

# ---- Figure 1: levels ----
plt.figure(figsize=(11.5, 3.6))
plt.plot(y.index, y.values, linewidth=1.4)
plt.title("Swap spread (10Y EUR) — level")
plt.ylabel("Level")
plt.xlabel(f"Date   (Sample: {sample_str})")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

# ---- Figure 2: first difference ----
plt.figure(figsize=(11.5, 3.6))
plt.plot(dy.index, dy.values, linewidth=1.2)
plt.title("Swap spread (10Y EUR) — first difference (weekly Δ)")
plt.ylabel("Δ level")
plt.xlabel(f"Date   (Sample: {sample_str})")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()


# In[4]:


# ===================== Summary table for data =====================
# Requires: pandas, numpy, scipy  (pip install pandas numpy scipy)

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# ---- CONFIG ----
from pathlib import Path

# Project root (auto-detect, works on any machine)
ROOT = Path(__file__).resolve().parents[1]

# Data locations
DATA_DIR = ROOT / "data"
DATA_SAMPLE = ROOT / "data_sample" / "sample.csv"

# Output folders
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
DATE_COL      = "Date"
TARGET_COL    = "Swap Spread"
EXOG_COLS     = [
    "Credit_Risk", "ECBrate", "German_CPI YoY", "Euribor-OIS",
    "Yield_Slope", "VSTOXX", "German_UnemploymentRate"
]
VAR_ORDER = [TARGET_COL] + EXOG_COLS

# Optional: show each series’ “native” frequency (for the LaTeX table notes)
FREQ_MAP = {
    "German_CPI YoY": "Monthly",
    "German_UnemploymentRate": "Monthly",
    # others are weekly in your dataset
}

# ---- Load & prep ----
df = pd.read_excel(PATH_TO_EXCEL)

if DATE_COL in df.columns:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

# keep only columns we want (present in file)
cols = [c for c in VAR_ORDER if c in df.columns]
data = df[cols].copy()

# ---- Summary function ----
def summarize(series: pd.Series, total_rows: int) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").dropna()
    n_nonmiss = int(x.size)
    n_miss = int(total_rows - n_nonmiss)
    miss_pct = 100.0 * n_miss / max(1, total_rows)

    # Guard against empty series
    if n_nonmiss == 0:
        return pd.Series({
            "N": 0, "Mean": np.nan, "Std": np.nan, "Min": np.nan,
            "P25": np.nan, "Median": np.nan, "P75": np.nan, "Max": np.nan,
            "Skew": np.nan, "Ex.Kurt": np.nan, "Missing": n_miss,
            "Missing %": miss_pct
        })

    return pd.Series({
        "N": n_nonmiss,
        "Mean": x.mean(),
        "Std": x.std(ddof=1),
        "Min": x.min(),
        "P25": x.quantile(0.25),
        "Median": x.median(),
        "P75": x.quantile(0.75),
        "Max": x.max(),
        "Skew": skew(x, bias=False),
        "Ex.Kurt": kurtosis(x, fisher=True, bias=False),  # excess kurtosis
        "Missing": n_miss,
        "Missing %": miss_pct
    })

# ---- Build table ----
total_rows = len(data)
summary_df = data.apply(lambda s: summarize(s, total_rows), axis=0).T

# Optional: add a Frequency column (for information only)
summary_df.insert(0, "Frequency", [FREQ_MAP.get(var, "Weekly") for var in summary_df.index])

# Round for display
round_cols = ["Mean","Std","Min","P25","Median","P75","Max","Skew","Ex.Kurt","Missing %"]
summary_display = summary_df.copy()
summary_display[round_cols] = summary_display[round_cols].round(3)
summary_display["N"] = summary_display["N"].astype(int)
summary_display["Missing"] = summary_display["Missing"].astype(int)

print("\nSummary statistics (levels):")
print(summary_display)

# ---- LaTeX (booktabs) ----
latex = summary_display.to_latex(
    index=True, escape=True,  # escape underscores in names
    column_format="l l r r r r r r r r r r",
    caption="Summary statistics for the swap spread and exogenous variables (levels).",
    label="tab:data_summary",
    na_rep="",
    bold_rows=False
)
print("\n% ---------- LaTeX table ----------")
print(latex)

# Optionally save:
# with open("table_data_summary.tex", "w") as f:
#     f.write(latex)


# In[5]:


# ================= Summary table with ADF-based Stationarity =================
# - Descriptives on LEVELS
# - Stationarity decided by ADF: first with constant ('c'); if not rejected,
#   try constant+trend ('ct'); else Non-stationary.

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# ---- CONFIG ----
from pathlib import Path

# Project root (auto-detect, works on any machine)
ROOT = Path(__file__).resolve().parents[1]

# Data locations
DATA_DIR = ROOT / "data"
DATA_SAMPLE = ROOT / "data_sample" / "sample.csv"

# Output folders
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
DATE_COL      = "Date"
TARGET_COL    = "Swap Spread"
EXOG_VARS     = [
    "Credit_Risk", "ECBrate", "German_CPI YoY", "Euribor-OIS",
    "Yield_Slope", "VSTOXX", "German_UnemploymentRate"
]
FREQ_MAP = {  # original data frequency
    "German_CPI YoY": "Monthly",
    "German_UnemploymentRate": "Monthly",
    TARGET_COL: "Weekly",
    "Credit_Risk": "Weekly",
    "ECBrate": "Weekly",
    "Euribor-OIS": "Weekly",
    "Yield_Slope": "Weekly",
    "VSTOXX": "Weekly",
}

# ---- Helpers ----
def adf_stationarity(series: pd.Series):
    x = series.dropna().astype(float).values
    if x.size < 20:
        return "Insufficient data", np.nan, np.nan
    # ADF with constant
    try:
        stat_c = adfuller(x, regression='c', autolag='AIC')
        p_c = stat_c[1]
    except Exception:
        p_c = np.nan
    if pd.notna(p_c) and p_c < 0.05:
        return "Stationary (c)", p_c, np.nan
    # ADF with constant + trend
    try:
        stat_ct = adfuller(x, regression='ct', autolag='AIC')
        p_ct = stat_ct[1]
    except Exception:
        p_ct = np.nan
    if pd.notna(p_ct) and p_ct < 0.05:
        return "Stationary (c+t)", p_c, p_ct
    return "Non-stationary", p_c, p_ct

def fmt(x, k=3):
    return f"{x:.{k}f}" if pd.notna(x) else ""

# ---- Load ----
df = pd.read_excel(PATH_TO_EXCEL)
if DATE_COL in df.columns:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).set_index(DATE_COL)

vars_all = [TARGET_COL] + EXOG_VARS
vars_present = [v for v in vars_all if v in df.columns]

rows = []
for v in vars_present:
    s = df[v].astype(float)
    N   = int(s.dropna().shape[0])
    mu  = s.mean()
    sd  = s.std(ddof=1)
    q1, med, q3 = s.quantile([0.25, 0.50, 0.75])
    vmin, vmax  = s.min(), s.max()
    skew = s.skew()
    kurt = s.kurt()  # excess kurtosis

    label, p_c, p_ct = adf_stationarity(s)
    rows.append({
        "Variable": v,
        "Frequency": FREQ_MAP.get(v, "Weekly"),
        "N": N,
        "Mean": mu, "Std. Dev.": sd,
        "Min": vmin, "Q1": q1, "Median": med, "Q3": q3, "Max": vmax,
        "Skew": skew, "Excess Kurt.": kurt,
        "Stationarity": label
    })

tab = pd.DataFrame(rows)

# Pretty preview
preview_cols = ["Variable","Frequency","N","Mean","Std. Dev.","Min","Q1","Median","Q3","Max","Skew","Excess Kurt.","Stationarity"]
print(tab[preview_cols].to_string(index=False,
      formatters={
          "Mean": fmt, "Std. Dev.": fmt, "Min": fmt, "Q1": fmt,
          "Median": fmt, "Q3": fmt, "Max": fmt, "Skew": fmt, "Excess Kurt.": fmt
      }))

# ---- LaTeX output ----
latex_tab = tab[preview_cols].to_latex(
    index=False,
    escape=False,
    float_format=lambda x: f"{x:.3f}",
    column_format="l l r r r r r r r r r r l",
    caption="Summary statistics and stationarity (ADF) for swap spread and exogenous variables (levels).",
    label="tab:data_summary_stationarity",
    bold_rows=False
).replace("toprule", "toprule").replace("midrule", "midrule").replace("bottomrule", "bottomrule")

# Add booktabs note block manually if you like:
notes = (
    "\\begin{flushleft}\\footnotesize "
    "Notes: Stationarity is assessed with the Augmented Dickey–Fuller test on levels. "
    "'Stationary (c)' indicates rejection at 5\\% with an intercept; "
    "'Stationary (c+t)' indicates rejection with intercept and linear trend; "
    "'Non-stationary' indicates failure to reject with either specification. "
    "Excess kurtosis is reported (zero for normal).\\end{flushleft}"
)
latex_full = latex_tab + "\n" + notes

print("\n" + latex_full)

