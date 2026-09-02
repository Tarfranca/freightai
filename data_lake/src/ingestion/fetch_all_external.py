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


def write_empty_csv(path: Path, columns=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def fetch_yahoo(ticker: str, start: str, end: str, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if data is None or data.empty:
            write_empty_csv(output_path)
            return {
                "status": "failed",
                "rows": 0,
                "date_range": f"{start} to {end}",
                "path": str(output_path),
                "message": f"No data returned for {ticker}.",
            }

        if isinstance(data.columns, pd.MultiIndex):
            data = data.copy()
            data.columns = [
                col[0] if isinstance(col, tuple) and len(col) > 1 else str(col)
                for col in data.columns
            ]

        data = data.reset_index()
        if "Date" not in data.columns:
            data = data.rename(columns={data.columns[0]: "Date"})
        data.columns = [str(col).strip() for col in data.columns]
        data.to_csv(output_path, index=False)

        date_range = f"{data['Date'].min().date()} to {data['Date'].max().date()}"
        return {
            "status": "success",
            "rows": len(data),
            "date_range": date_range,
            "path": str(output_path),
            "message": "OK",
        }
    except Exception as exc:  # pragma: no cover
        write_empty_csv(output_path)
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
        write_empty_csv(output_path, columns=["Date", "Index", "Source"])
        return {
            "status": "failed",
            "rows": 0,
            "date_range": "n/a",
            "path": str(output_path),
            "message": str(exc),
        }


def fetch_fbx_investing(output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        "https://www.investing.com/indices/freightos-baltic-index-historical-data",
        "https://www.investing.com/indices/freightos-baltic-index",
    ]
    for url in urls:
        try:
            tables = pd.read_html(url)
            if not tables:
                continue
            table = tables[0]
            if table.empty:
                continue
            table.to_csv(output_path, index=False)
            return {
                "status": "success",
                "rows": len(table),
                "date_range": "n/a",
                "path": str(output_path),
                "message": f"Scraped FBX from {url}",
            }
        except Exception:
            continue

    write_empty_csv(output_path, columns=["Date", "Index", "Source"])
    return {
        "status": "failed",
        "rows": 0,
        "date_range": "n/a",
        "path": str(output_path),
        "message": "Could not fetch FBX from Investing.com.",
    }


def fetch_bdi_market(output_path: Path, start_date: str, end_date: str) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        "https://markets.businessinsider.com/commodities/baltic-dry-index",
        "https://markets.businessinsider.com/commodities/baltic-dry-index?type=historical",
    ]
    for url in urls:
        try:
            tables = pd.read_html(url)
            if not tables:
                continue
            table = tables[0]
            if table.empty:
                continue
            table.to_csv(output_path, index=False)
            return {
                "status": "success",
                "rows": len(table),
                "date_range": f"{start_date} to {end_date}",
                "path": str(output_path),
                "message": f"Scraped BDI from {url}",
            }
        except Exception:
            continue

    fallback = fetch_yahoo("BDRY", start_date, end_date, output_path)
    if fallback["status"] == "success":
        fallback["message"] = "BDRY fallback used after web fetch failed."
        return fallback

    write_empty_csv(output_path)
    return {
        "status": "failed",
        "rows": 0,
        "date_range": f"{start_date} to {end_date}",
        "path": str(output_path),
        "message": "Could not fetch BDI from web sources and fallback BDRY returned no data.",
    }


def main() -> None:
    ensure_yfinance()

    start_date = "2024-01-01"
    end_date = "2026-01-03"

    sources = [
        {"name": "USD/BRL", "ticker": "BRL=X", "output": RAW_DIR / "macro_finance" / "usd_brl_daily.csv"},
        {"name": "US Dollar Index", "ticker": "DX-Y.NYB", "output": RAW_DIR / "macro_finance" / "dxy_daily.csv"},
        {"name": "US 10Y Treasury Yield", "ticker": "^TNX", "output": RAW_DIR / "macro_finance" / "us_10y_yield.csv"},
        {"name": "Brent Oil", "ticker": "BZ=F", "output": RAW_DIR / "energy" / "brent_daily.csv"},
        {"name": "Crude Oil WTI", "ticker": "CL=F", "output": RAW_DIR / "energy" / "wti_daily.csv"},
        {"name": "S&P 500", "ticker": "^GSPC", "output": RAW_DIR / "market" / "sp500_daily.csv"},
        {"name": "VIX", "ticker": "^VIX", "output": RAW_DIR / "market" / "vix_daily.csv"},
        {"name": "Baltic Dry Index proxy ETF", "ticker": "BDRY", "output": RAW_DIR / "market" / "bdry_etf.csv"},
        {"name": "ZIM Integrated Shipping", "ticker": "ZIM", "output": RAW_DIR / "market" / "zim_daily.csv"},
        {"name": "Maersk", "ticker": "MAERSK-B.CO", "output": RAW_DIR / "market" / "maersk_daily.csv"},
        {"name": "Gold", "ticker": "GC=F", "output": RAW_DIR / "commodity" / "gold_daily.csv"},
    ]

    results = []
    for item in sources:
        result = fetch_yahoo(item["ticker"], start_date, end_date, item["output"])
        results.append({"source": item["name"], **result})
        print(f"{item['name']}: rows={result['rows']}, date_range={result['date_range']}, status={result['status']}")

    fbx_result = fetch_fbx_investing(RAW_DIR / "market" / "fbx_daily.csv")
    results.append({"source": "Freightos Baltic Index", **fbx_result})
    print(f"Freightos Baltic Index: rows={fbx_result['rows']}, date_range={fbx_result['date_range']}, status={fbx_result['status']}")

    bdi_result = fetch_bdi_market(RAW_DIR / "market" / "bdi_daily.csv", start_date, end_date)
    results.append({"source": "Baltic Dry Index", **bdi_result})
    print(f"Baltic Dry Index: rows={bdi_result['rows']}, date_range={bdi_result['date_range']}, status={bdi_result['status']}")

    drewry_result = fetch_drewry(RAW_DIR / "market" / "drewry_wci.csv")
    results.append({"source": "Drewry WCI", **drewry_result})
    print(f"Drewry WCI: rows={drewry_result['rows']}, date_range={drewry_result['date_range']}, status={drewry_result['status']}")

    print("\nSummary:")
    for item in results:
        print(f"- {item['source']}: {item['status']} | rows={item['rows']} | {item['date_range']} | {item['message']}")


if __name__ == "__main__":
    main()
