"""Utility helpers for loading price history with offline fallbacks."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:  # pragma: no cover - optional dependency
    import yfinance as yf  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yf = None

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SAMPLE_MULTI_ASSET = DATA_DIR / "sample_multi_asset_prices.csv"
TICKERS = ["SPY", "TLT", "GLD"]


def _load_sample_prices() -> pd.DataFrame:
    if not SAMPLE_MULTI_ASSET.exists():
        raise FileNotFoundError(
            "Bundled sample price history is missing; expected "
            f"{SAMPLE_MULTI_ASSET}"
        )
    df = pd.read_csv(SAMPLE_MULTI_ASSET, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def _filter_start(df: pd.DataFrame, start_date: Optional[str]) -> pd.DataFrame:
    if start_date:
        start_ts = pd.to_datetime(start_date)
        df = df.loc[df.index >= start_ts]
    return df


def _download_close_prices(tickers: list[str], start_date: str) -> Optional[pd.DataFrame]:
    if yf is None:
        return None

    try:
        raw = yf.download(
            " ".join(tickers),
            start=start_date,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"Warning: yfinance download failed ({exc}); falling back to sample data.")
        return None

    if raw.empty:
        return None

    if "Close" in raw.columns:
        df_close = raw["Close"].copy()
    elif isinstance(raw.columns, pd.MultiIndex):
        df_close = raw.xs("Close", axis=1, level=1).copy()
    else:
        df_close = raw.copy()

    if isinstance(df_close, pd.Series):
        df_close = df_close.to_frame()

    df_close = df_close.dropna(how="any")
    missing = [t for t in tickers if t not in df_close.columns]
    if missing:
        return None

    df_close = df_close[tickers]
    df_close.index = pd.to_datetime(df_close.index)
    return df_close


def load_multi_asset_history(start_date: str) -> pd.DataFrame:
    """Load SPY, TLT, and GLD close prices with offline fallback."""
    df = _download_close_prices(TICKERS, start_date)
    if df is None:
        print("Using bundled sample multi-asset history (offline fallback).")
        df = _load_sample_prices()[TICKERS]
    return _filter_start(df, start_date)


def load_spy_history(start_date: str) -> pd.DataFrame:
    """Load SPY close prices with offline fallback."""
    df = _download_close_prices(["SPY"], start_date)
    if df is None:
        print("Using bundled sample SPY history (offline fallback).")
        df = _load_sample_prices()[["SPY"]]
    return _filter_start(df, start_date)
