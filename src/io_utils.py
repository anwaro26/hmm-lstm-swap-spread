from pathlib import Path
import pandas as pd

def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"Unsupported file type: {path}")