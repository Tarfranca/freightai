"""Fetches economic indices and port climate data for the FreightAI data lake.

Sources (Phase 4):
  1. FRED — US GDP (GDPC1) and CPI (CPIAUCSL) as CSV. NAPM (ISM Manufacturing
     PMI) is included per spec but ISM discontinued public redistribution of
     that series through FRED some years ago (confirmed 404 below) — logged
     and skipped rather than faked.
  2. World Bank — GDP growth and Logistics Performance Index, by country/year,
     fetched in a single page (per_page large enough to cover the whole series).
  3. IMF WEO (via the public Datamapper API) — real GDP growth, extracting the
     world aggregate ("WEOWORLD") alongside every country's series.
  4. Open-Meteo — daily windspeed/precipitation/weather code for 6 major ports,
     last 30 days, no API key required.
  5. Panama Canal ENSO risk — a small derived table computed locally from the
     ONI data already fetched in Phase 3 (no network call), per the thresholds
     given: ONI > 0.5 -> drought risk HIGH, ONI < -0.5 -> flood risk MEDIUM,
     else NORMAL.
"""
import subprocess
import sys
from pathlib import Path

try:
    import pandas as pd
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "requests"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import pandas as pd
    import requests

BASE_DIR = Path(__file__).resolve().parents[2]
MACRO_DIR = BASE_DIR / "raw" / "macro_finance"
CLIMATE_DIR = BASE_DIR / "raw" / "climate"
RISK_DIR = BASE_DIR / "raw" / "risk"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FreightAI-DataLake/1.0)"}

FRED_SERIES = ["GDPC1", "CPIAUCSL", "NAPM"]
WORLD_BANK_INDICATORS = ["NY.GDP.MKTP.KD.ZG", "LP.LPI.OVRL.XQ"]
PORTS = {
    "Shanghai": (31.23, 121.47),
    "Rotterdam": (51.90, 4.48),
    "Santos": (-23.96, -46.33),
    "Los Angeles": (33.74, -118.27),
    "Hamburg": (53.55, 9.99),
    "Singapore": (1.29, 103.85),
}


def fetch_fred() -> int:
    total_rows = 0
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    for series_id in FRED_SERIES:
        out_path = MACRO_DIR / f"fred_{series_id}.csv"
        try:
            # No custom User-Agent here on purpose: FRED's WAF blocks/hangs on it
            # (verified — the same request succeeds instantly with the plain
            # requests-library default UA and fails with any custom one tried).
            response = requests.get(
                "https://fred.stlouisfed.org/graph/fredgraph.csv",
                params={"id": series_id}, timeout=20,
            )
            response.raise_for_status()
            if not response.text.lstrip().startswith("observation_date"):
                raise ValueError("response wasn't a CSV (series likely discontinued/renamed at FRED)")
            out_path.write_text(response.text, encoding="utf-8")
            df = pd.read_csv(out_path)
            print(f"[FRED {series_id}] OK — {len(df)} rows ({df.iloc[0, 0]} to {df.iloc[-1, 0]}) -> {out_path}")
            total_rows += len(df)
        except Exception as exc:
            print(f"[FRED {series_id}] FAILED — {exc}")
    return total_rows


def fetch_world_bank() -> int:
    total_rows = 0
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    for indicator in WORLD_BANK_INDICATORS:
        out_path = MACRO_DIR / f"worldbank_{indicator}.csv"
        try:
            response = requests.get(
                f"https://api.worldbank.org/v2/country/all/indicator/{indicator}",
                params={"format": "json", "per_page": 20000}, headers=HEADERS, timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            records = payload[1] if len(payload) > 1 and payload[1] else []
            if not records:
                raise ValueError("World Bank API returned no records")

            rows = [{
                "country": rec["country"]["value"],
                "country_code": rec["countryiso3code"],
                "year": rec["date"],
                "value": rec["value"],
            } for rec in records]
            df = pd.DataFrame(rows)
            df.to_csv(out_path, index=False)
            non_null = df["value"].notna().sum()
            print(f"[World Bank {indicator}] OK — {len(df)} rows, {non_null} with a value "
                  f"-> {out_path}")
            total_rows += len(df)
        except Exception as exc:
            print(f"[World Bank {indicator}] FAILED — {exc}")
    return total_rows


def fetch_imf_gdp_growth() -> int:
    out_path = MACRO_DIR / "imf_gdp_growth.csv"
    try:
        # Same deal as FRED: the IMF's edge/WAF rejects a custom User-Agent here.
        response = requests.get(
            "https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH",
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        series = payload["values"]["NGDP_RPCH"]

        rows = []
        for country_code, year_values in series.items():
            for year, value in year_values.items():
                rows.append({"country_code": country_code, "year": int(year), "gdp_growth_pct": value})
        df = pd.DataFrame(rows).sort_values(["country_code", "year"])
        MACRO_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        world = df[df["country_code"] == "WEOWORLD"].sort_values("year")
        latest_world = world.iloc[-1] if not world.empty else None
        world_note = (f", world forecast {latest_world['year']}: {latest_world['gdp_growth_pct']:+.1f}%"
                       if latest_world is not None else "")
        print(f"[IMF WEO GDP Growth] OK — {len(df)} rows across {df['country_code'].nunique()} "
              f"countries/aggregates{world_note} -> {out_path}")
        return len(df)
    except Exception as exc:
        print(f"[IMF WEO GDP Growth] FAILED — {exc}")
        return 0


def fetch_port_weather() -> int:
    out_path = CLIMATE_DIR / "port_weather.csv"
    frames = []
    for port_name, (lat, lon) in PORTS.items():
        try:
            response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "daily": "windspeed_10m_max,precipitation_sum,weathercode",
                    "past_days": 30, "forecast_days": 1, "timezone": "UTC",
                },
                headers=HEADERS, timeout=20,
            )
            response.raise_for_status()
            daily = response.json()["daily"]
            frame = pd.DataFrame({
                "port": port_name,
                "date": daily["time"],
                "windspeed_10m_max_kmh": daily["windspeed_10m_max"],
                "precipitation_sum_mm": daily["precipitation_sum"],
                "weathercode": daily["weathercode"],
            })
            frames.append(frame)
            print(f"[Open-Meteo] {port_name}: {len(frame)} days fetched")
        except Exception as exc:
            print(f"[Open-Meteo] {port_name} FAILED — {exc}")

    if not frames:
        print("[Open-Meteo] FAILED — no ports returned data")
        return 0

    combined = pd.concat(frames, ignore_index=True)
    CLIMATE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"[Open-Meteo] OK — {len(combined)} total rows across {len(frames)} ports -> {out_path}")
    return len(combined)


def compute_panama_enso_risk() -> int:
    out_path = CLIMATE_DIR / "panama_enso_risk.csv"
    oni_path = RISK_DIR / "oni_enso.csv"
    try:
        if not oni_path.exists():
            raise FileNotFoundError(f"{oni_path} not found — run fetch_risk_indices.py (Phase 3) first")

        oni = pd.read_csv(oni_path)
        current = oni.iloc[-1]
        oni_value = float(current["oni_value"])

        if oni_value > 0.5:
            drought_risk, draft_restriction = "HIGH", "Likely — El Nino historically lowers Gatun Lake levels"
        elif oni_value < -0.5:
            drought_risk, draft_restriction = "LOW (flood risk MEDIUM)", "Unlikely — La Nina brings above-average rainfall"
        else:
            drought_risk, draft_restriction = "NORMAL", "Unlikely under neutral conditions"

        row = {
            "computed_at_season": current["season"],
            "computed_at_year": int(current["year"]),
            "oni_value": oni_value,
            "enso_phase": current["enso_phase"],
            "panama_drought_risk": drought_risk,
            "panama_draft_restriction_outlook": draft_restriction,
        }
        df = pd.DataFrame([row])
        CLIMATE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[Panama ENSO Risk] OK — {current['enso_phase']} ({oni_value:+.2f}) -> "
              f"drought risk {drought_risk} -> {out_path}")
        return 1
    except Exception as exc:
        print(f"[Panama ENSO Risk] FAILED — {exc}")
        return 0


def main() -> None:
    print("=== Phase 4: Economic Indices & Port Climate Ingestion ===")

    fred_rows = fetch_fred()
    wb_rows = fetch_world_bank()
    imf_rows = fetch_imf_gdp_growth()
    weather_rows = fetch_port_weather()
    panama_rows = compute_panama_enso_risk()

    print()
    print("=== Summary ===")
    print(f"FRED (3 series attempted): {fred_rows} rows fetched")
    print(f"World Bank (2 indicators): {wb_rows} rows fetched")
    print(f"IMF WEO GDP Growth:        {imf_rows} rows fetched")
    print(f"Open-Meteo port weather:   {weather_rows} rows fetched")
    print(f"Panama ENSO risk summary:  {panama_rows} rows fetched")


if __name__ == "__main__":
    main()
