import os
import pandas as pd
from src.data.generate import FILES

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATE_COLS = ["sale_date", "order_date", "expected_date", "received_date", "date"]


def load_bundled(data_dir: str = None) -> dict:
    """Load bundled demo CSVs (regenerates them deterministically if missing)."""
    data_dir = data_dir or os.path.join(ROOT, "data")
    if not os.path.exists(os.path.join(data_dir, FILES["sales"])):
        from src.data.generate import write_csvs
        write_csvs(data_dir)
    out = {}
    for k, f in FILES.items():
        df = pd.read_csv(os.path.join(data_dir, f))
        for c in DATE_COLS:
            if c in df.columns: df[c] = pd.to_datetime(df[c])
        out[k] = df
    return out
