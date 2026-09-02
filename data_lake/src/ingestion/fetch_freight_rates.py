"""Fetches freight rate benchmarks for the FreightAI data lake.

Sources (Phase 2 — scoped to what's actually reachable and free):
  1. SCFI (Shanghai Containerized Freight Index) — real live JSON endpoint on the
     Shanghai Shipping Exchange site. Only the Comprehensive Index is reliably
     populated in the public (non-subscriber) response; per-route legs are mostly
     null unless you're logged in, so those are saved as NaN rather than guessed.
  2. Drewry WCI (World Container Index) — the public page doesn't expose a JSON
     API, but the current headline figure is embedded in the page's <meta
     name="description"> tag (e.g. "27 Aug 2026: ... decreased 1% to $4,473 per
     40ft container"), which is scraped with a regex.
  3. Air freight index — TAC Index / Baltic Air Freight have no free public feed,
     so this uses ATSG, FDX, and UPS (three companies whose stock price reacts
     directly to air cargo demand/pricing) via yfinance as a proxy.

MDIC and ANTAQ are intentionally skipped here (large bulk downloads / dead
source URL) — see the project notes for follow-up.
"""
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import pandas as pd
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "requests"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import pandas as pd
    import requests

try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import yfinance as yf

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "raw" / "freight_rates"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FreightAI-DataLake/1.0)"}

SCFI_ROUTE_MAP = {
    "Comprehensive Index": "Comprehensive Index",
    "Europe 20ft (Base port)": "Europe",
    "Mediterranean 20ft (Base port)": "Mediterranean",
    "USWC 40ft (Base port)": "US West Coast",
    "USEC 40ft (Base port)": "US East Coast",
    "Persian Gulf and Red Sea 20ft (Dubai)": "Persian Gulf",
    "West Japan 20ft (Base port)": "Japan",
    "East Japan 20ft (Base port)": "Japan",
    "Southeast Asia 20ft (Singapore)": "Southeast Asia",
    "South America 20ft (Santos)": "South America",
    "Australia/New Zealand 20ft (Melbourne)": "Australia",
}


def append_or_create(path: Path, new_row: dict, dedupe_keys: list[str]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([new_row])
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=dedupe_keys, keep="last")
    else:
        combined = new_df
    combined.to_csv(path, index=False)
    return combined


def fetch_scfi() -> int:
    """Real SCFI data from the Shanghai Shipping Exchange public JSON endpoint."""
    out_path = OUT_DIR / "scfi_weekly.csv"
    try:
        response = requests.get(
            "https://en.sse.net.cn/currentIndex",
            params={"indexName": "scfi"},
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or {}
        lines = data.get("lineDataList") or []
        current_date = data.get("currentDate")

        rows = []
        for line in lines:
            props = line.get("properties") or {}
            route_en = props.get("lineName_EN")
            route = SCFI_ROUTE_MAP.get(route_en, route_en)
            current_value = line.get("currentContent")
            last_value = line.get("lastContent")
            week_change_pct = line.get("percentage")
            rows.append({
                "date": current_date,
                "route": route,
                "index_value": current_value,
                "previous_value": last_value,
                "week_change_pct": week_change_pct,
            })

        if not rows:
            raise ValueError("SCFI response contained no route lines")

        new_df = pd.DataFrame(rows)
        if out_path.exists():
            existing = pd.read_csv(out_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date", "route"], keep="last")
        else:
            combined = new_df
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)

        populated = new_df["index_value"].notna().sum()
        print(f"[SCFI] OK — {len(new_df)} route rows fetched for {current_date} "
              f"({populated} with a live value; the rest are subscriber-only on the public feed). "
              f"File now has {len(combined)} rows total -> {out_path}")
        return len(new_df)
    except Exception as exc:
        print(f"[SCFI] FAILED — {exc}")
        return 0


def fetch_drewry_wci() -> int:
    """Drewry WCI headline figure, scraped from the page's meta description."""
    out_path = OUT_DIR / "drewry_wci_weekly.csv"
    url = "https://www.drewry.co.uk/supply-chain-advisors/supply-chain-expertise/world-container-index-assessed-by-drewry"
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        html = response.text

        desc_match = re.search(r'<meta name="description" content="([^"]+)"', html)
        if not desc_match:
            raise ValueError("Could not find the meta description tag on the Drewry page")
        description = desc_match.group(1)
        description = (description.replace("&#x3a;", ":").replace("&rsquo;", "'")
                        .replace("&#x25;", "%").replace("&#x24;", "$")
                        .replace("&ndash;", "-").replace("&#x28;", "(").replace("&#x29;", ")"))

        date_match = re.search(r"(\d{1,2}\s+\w{3}\s*\d{0,4})", description)
        value_match = re.search(r"to\s*\$?([\d,]+)\s*per\s*40ft", description, re.IGNORECASE)
        change_match = re.search(r"(increased|decreased)\s+(\d+)%", description, re.IGNORECASE)
        if not value_match:
            raise ValueError(f"Could not parse the WCI figure out of: {description!r}")

        index_value = float(value_match.group(1).replace(",", ""))
        week_change_pct = None
        if change_match:
            sign = 1 if change_match.group(1).lower() == "increased" else -1
            week_change_pct = sign * float(change_match.group(2))

        raw_date = date_match.group(1).strip() if date_match else ""
        parsed_date = None
        for fmt in ("%d %b %Y", "%d %b"):
            try:
                parsed = datetime.strptime(raw_date, fmt)
                if fmt == "%d %b":
                    parsed = parsed.replace(year=datetime.now(timezone.utc).year)
                parsed_date = parsed.strftime("%Y-%m-%d")
                break
            except ValueError:
                continue

        row = {
            "date": parsed_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "index_value_usd_per_40ft": index_value,
            "week_change_pct": week_change_pct,
            "source_text": description,
        }
        combined = append_or_create(out_path, row, dedupe_keys=["date"])
        print(f"[Drewry WCI] OK — ${index_value:,.0f}/40ft on {row['date']} "
              f"({week_change_pct:+.0f}% w/w). File now has {len(combined)} rows total -> {out_path}")
        return 1
    except Exception as exc:
        print(f"[Drewry WCI] FAILED — {exc}")
        return 0


def fetch_air_freight_proxy() -> int:
    """Air cargo demand/pricing proxy via publicly traded air-freight carriers."""
    out_path = OUT_DIR / "air_freight_index.csv"
    tickers = ["ATSG", "FDX", "UPS"]
    try:
        data = yf.download(tickers, period="2y", auto_adjust=False, progress=False, group_by="ticker")
        if data is None or data.empty:
            raise ValueError("yfinance returned no data for ATSG/FDX/UPS")

        frames = []
        for ticker in tickers:
            try:
                close = data[ticker]["Close"].dropna()
            except (KeyError, TypeError):
                print(f"[Air Freight Proxy]   ... no data returned for {ticker}, skipping it")
                continue
            frame = close.reset_index()
            frame.columns = ["date", "close"]
            frame["ticker"] = ticker
            frames.append(frame)

        if not frames:
            raise ValueError("None of ATSG/FDX/UPS returned usable data")

        combined = pd.concat(frames, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")

        # Composite proxy index: equal-weighted average of the three tickers'
        # z-scored prices, rescaled to start at 1000 (same convention as the
        # platform's other index-style series).
        pivot = combined.pivot(index="date", columns="ticker", values="close").sort_index()
        zscored = (pivot - pivot.mean()) / pivot.std()
        composite = zscored.mean(axis=1)
        composite_index = (composite - composite.iloc[0]) * 100 + 1000

        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)

        composite_path = OUT_DIR / "air_freight_index_composite.csv"
        composite.rename("composite_score").to_frame().assign(
            index_value=composite_index
        ).reset_index().to_csv(composite_path, index=False)

        print(f"[Air Freight Proxy] OK — {len(combined)} price rows across {combined['ticker'].nunique()} "
              f"tickers ({', '.join(sorted(combined['ticker'].unique()))}) -> {out_path}")
        print(f"[Air Freight Proxy]   composite index saved -> {composite_path}")
        return len(combined)
    except Exception as exc:
        print(f"[Air Freight Proxy] FAILED — {exc}")
        return 0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Phase 2: Freight Rates Ingestion ===")

    scfi_rows = fetch_scfi()
    drewry_rows = fetch_drewry_wci()
    air_rows = fetch_air_freight_proxy()

    print()
    print("[MDIC] SKIPPED — deferred (290MB combined bulk download), documented as future work")
    print("[ANTAQ] SKIPPED — spec URL returns 404 (page moved/dead), no working alternative found yet")

    print()
    print("=== Summary ===")
    print(f"SCFI:              {scfi_rows} rows fetched")
    print(f"Drewry WCI:        {drewry_rows} rows fetched")
    print(f"Air Freight Proxy: {air_rows} rows fetched")


if __name__ == "__main__":
    main()
