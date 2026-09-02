import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "raw"


def ensure_yfinance() -> None:
    try:
        __import__("yfinance")
    except ImportError:
        print("Installing yfinance...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "yfinance"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


def fetch_yahoo(ticker: str, start: str, end: str, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if data.empty:
            return {
                "status": "failed",
                "rows": 0,
                "date_range": f"{start} to {end}",
                "path": str(output_path),
                "message": f"No data returned for {ticker}.",
            }
        data = data.reset_index()
        data.columns = [str(col).strip() for col in data.columns]
        data.to_csv(output_path, index=False)
        return {
            "status": "success",
            "rows": len(data),
            "date_range": f"{data['Date'].min().date()} to {data['Date'].max().date()}",
            "path": str(output_path),
            "message": "OK",
        }
    except Exception as exc:  # pragma: no cover
        return {
            "status": "failed",
            "rows": 0,
            "date_range": f"{start} to {end}",
            "path": str(output_path),
            "message": str(exc),
        }


def fetch_drewry(output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://www.drewry.co.uk/supply-chain-advisors/supply-chain-expertise/world-container-index-assessed-by-drewry"
    try:
        tables = pd.read_html(url)
        if not tables:
            raise ValueError("No HTML tables found on page.")
        table = tables[0]
        table.to_csv(output_path, index=False)
        return {
            "status": "success",
            "rows": len(table),
            "date_range": "n/a",
            "path": str(output_path),
            "message": "Scraped first HTML table from page.",
        }
    except Exception as exc:  # pragma: no cover
        return {
            "status": "failed",
            "rows": 0,
            "date_range": "n/a",
            "path": str(output_path),
            "message": str(exc),
        }


def main() -> None:
    ensure_yfinance()

    start_date = "2024-01-01"
    end_date = "2026-01-03"

    sources = [
        {
            "name": "USD/BRL",
            "ticker": "BRL=X",
            "output": RAW_DIR / "macro_finance" / "usd_brl_daily.csv",
        },
        {
            "name": "Brent Oil",
            "ticker": "BZ=F",
            "output": RAW_DIR / "energy" / "brent_daily.csv",
        },
        {
            "name": "Baltic Dry Index",
            "ticker": "^BDI",
            "output": RAW_DIR / "market" / "bdi_daily.csv",
        },
        {
            "name": "Gold",
            "ticker": "GC=F",
            "output": RAW_DIR / "commodity" / "gold_daily.csv",
        },
    ]

    results = []
    for item in sources:
        result = fetch_yahoo(item["ticker"], start_date, end_date, item["output"])
        results.append({"source": item["name"], **result})
        print(
            f"{item['name']}: rows={result['rows']}, date_range={result['date_range']}, status={result['status']}"
        )

    drewry_result = fetch_drewry(RAW_DIR / "market" / "drewry_wci.csv")
    results.append({"source": "Drewry WCI", **drewry_result})
    print(
        f"Drewry WCI: rows={drewry_result['rows']}, date_range={drewry_result['date_range']}, status={drewry_result['status']}"
    )

    print("\nSummary:")
    for item in results:
        print(f"- {item['source']}: {item['status']} | rows={item['rows']} | {item['date_range']} | {item['message']}")


if __name__ == "__main__":
    main()
