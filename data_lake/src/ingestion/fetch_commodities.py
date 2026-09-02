"""Fetches commodity, shipping-stock, and macro price series for the FreightAI
data lake via yfinance, then builds a date-aligned merged master file.
"""
import re
import subprocess
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import pandas as pd

try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import yfinance as yf

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "raw" / "commodities"
GOLD_PATH = BASE_DIR / "gold" / "commodities_master.csv"
START_DATE = "2020-01-01"

TICKERS = {
    "Energy": ["BZ=F", "CL=F", "NG=F", "HO=F"],
    "Metals": ["GC=F", "SI=F", "HG=F", "SLX"],
    "Agriculture": ["ZS=F", "ZC=F", "SB=F", "KC=F", "ZW=F"],
    "Shipping": ["ZIM", "MATX", "DAC", "SBLK", "BDRY"],
    "Macro": ["DX-Y.NYB", "EURUSD=X", "BRL=X", "CNY=X", "JPY=X", "^VIX", "^GSPC", "^TNX"],
}


def clean_ticker_name(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")


def fetch_one(ticker: str) -> dict:
    out_path = OUT_DIR / f"{clean_ticker_name(ticker)}.csv"
    try:
        data = yf.download(ticker, start=START_DATE, auto_adjust=False, progress=False)
        if data is None or data.empty:
            return {"ticker": ticker, "status": "failed", "rows": 0, "date_range": "-", "reason": "no data returned"}

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        data = data.reset_index()
        data.columns = [str(c).strip() for c in data.columns]

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(out_path, index=False)

        date_range = f"{data['Date'].min().date()} to {data['Date'].max().date()}"
        return {"ticker": ticker, "status": "success", "rows": len(data), "date_range": date_range, "reason": None}
    except Exception as exc:
        return {"ticker": ticker, "status": "failed", "rows": 0, "date_range": "-", "reason": str(exc)}


def build_master(results: list[dict]) -> int:
    close_frames = []
    for result in results:
        if result["status"] != "success":
            continue
        ticker = result["ticker"]
        path = OUT_DIR / f"{clean_ticker_name(ticker)}.csv"
        df = pd.read_csv(path, parse_dates=["Date"])
        if "Close" not in df.columns:
            continue
        series = df.set_index("Date")["Close"].rename(ticker)
        close_frames.append(series)

    if not close_frames:
        return 0

    master = pd.concat(close_frames, axis=1, sort=True)
    master = master.asfreq("D")
    master = master.ffill(limit=3)
    master = master.dropna(how="all")

    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.reset_index().rename(columns={"Date": "date"}).to_csv(GOLD_PATH, index=False)
    return len(master)


def main() -> None:
    print("=== Phase 5: Commodities Ingestion ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for category, tickers in TICKERS.items():
        print(f"\n--- {category} ---")
        for ticker in tickers:
            result = fetch_one(ticker)
            all_results.append(result)
            if result["status"] == "success":
                print(f"[{ticker}] OK — {result['rows']} rows ({result['date_range']})")
            else:
                print(f"[{ticker}] FAILED — {result['reason']}")

    master_rows = build_master(all_results)

    print()
    print("=== Summary ===")
    print(f"{'Ticker':<12} {'Status':<10} {'Rows':>6}  Date Range")
    for result in all_results:
        print(f"{result['ticker']:<12} {result['status']:<10} {result['rows']:>6}  {result['date_range']}")

    success_count = sum(1 for r in all_results if r["status"] == "success")
    print()
    print(f"Total: {success_count}/{len(all_results)} tickers fetched successfully")
    print(f"Merged master file: {master_rows} aligned daily rows -> {GOLD_PATH}")


if __name__ == "__main__":
    main()
