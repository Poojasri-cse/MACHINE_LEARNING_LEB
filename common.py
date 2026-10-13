"""Shared helpers for Machine Learning Lab exercises."""
from pathlib import Path
import urllib.request
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_csv(filename, url, **kwargs):
    """Load a cached CSV or download it from a public URL."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    return pd.read_csv(path, **kwargs)
