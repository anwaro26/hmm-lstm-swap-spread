"""
Run the end-to-end pipeline on a small example dataset.

Portfolio entrypoint:
- loads sample data
- runs minimal validation
- prints a small summary so reviewers can see it works
"""
from pathlib import Path
import pandas as pd
from src.paths import DATA_SAMPLE, RESULTS_DIR

def main():
    sample_path = Path("data_sample") / "sample.csv"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Missing {sample_path}. Create data_sample/sample.csv first."
        )

    df = pd.read_csv(sample_path)

    required = {"date", "swap_spread"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sample data missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print("✅ Loaded sample dataset")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("Columns:", ", ".join(df.columns))
    print("\nHead:\n", df.head(3).to_string(index=False))

    print("\nNext: wire preprocessing.py -> hmm_lstm.py -> evaluate.py into this entrypoint.")

if __name__ == "__main__":
    main()