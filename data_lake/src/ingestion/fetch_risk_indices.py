"""Fetches geopolitical and climate risk indices for the FreightAI data lake.

Sources (Phase 3 — scoped to what's actually reachable and free):
  1. GPR (Geopolitical Risk Index, Caldara & Iacoviello) — monthly global index
     plus per-country columns, downloaded directly from Iacoviello's own site.
  2. WUI (World Uncertainty Index) — quarterly global index and regional
     breakdown, downloaded from the official WUI data file.
  3. ONI (Oceanic Nino Index / ENSO) — NOAA's plain-text seasonal SST anomaly
     table, used to flag El Nino / La Nina conditions (relevant to Panama Canal
     draft restrictions and Pacific storm patterns).
  4. OFAC SDN list — the Treasury's Specially Designated Nationals list,
     including the vessel-type entries relevant to shipping sanctions screening.

GDELT and ACLED are intentionally skipped — both were unreachable from this
environment during source discovery (GDELT: expired/parked domain; ACLED:
requires a registered API key we don't have) — documented as future work.
"""
import re
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

try:
    import openpyxl  # noqa: F401  (needed by pandas to read .xlsx)
    import xlrd  # noqa: F401  (needed by pandas to read legacy .xls)
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl", "xlrd"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "raw" / "risk"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FreightAI-DataLake/1.0)"}

SEASON_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def fetch_gpr() -> int:
    """Caldara & Iacoviello Geopolitical Risk Index — global, threat, act, and per-country."""
    out_path = OUT_DIR / "gpr_index.csv"
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        tmp_path = OUT_DIR / "_gpr_download.xls"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(response.content)

        raw = pd.read_excel(tmp_path)
        tmp_path.unlink(missing_ok=True)

        country_cols = [c for c in raw.columns if c.startswith("GPRC_")]
        keep = ["month", "GPR", "GPRT", "GPRA"] + country_cols
        df = raw[keep].copy()
        df = df.rename(columns={
            "month": "date", "GPR": "gpr_global", "GPRT": "gpr_threat", "GPRA": "gpr_act",
        })
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df.columns = [c.replace("GPRC_", "gpr_country_") if c.startswith("GPRC_") else c for c in df.columns]

        df.to_csv(out_path, index=False)
        print(f"[GPR] OK — {len(df)} monthly rows ({df['date'].min()} to {df['date'].max()}), "
              f"{len(country_cols)} per-country columns -> {out_path}")
        return len(df)
    except Exception as exc:
        print(f"[GPR] FAILED — {exc}")
        return 0


def fetch_wui() -> int:
    """World Uncertainty Index — global quarterly series plus regional breakdown."""
    out_path = OUT_DIR / "wui_index.csv"
    url = "https://worlduncertaintyindex.com/wp-content/uploads/2026/07/WUI_Data.xlsx"
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        tmp_path = OUT_DIR / "_wui_download.xlsx"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(response.content)

        global_df = pd.read_excel(tmp_path, sheet_name="F1", skiprows=2)
        global_df = global_df.iloc[:, :3]
        global_df.columns = ["quarter", "year", "wui_global"]
        global_df = global_df.dropna(subset=["quarter"])

        regional_df = pd.read_excel(tmp_path, sheet_name="T1")
        regional_df = regional_df.rename(columns={"Year": "quarter"})
        tmp_path.unlink(missing_ok=True)

        merged = global_df.merge(regional_df, on="quarter", how="left")
        merged.to_csv(out_path, index=False)
        print(f"[WUI] OK — {len(merged)} quarterly rows ({merged['quarter'].iloc[0]} to "
              f"{merged['quarter'].iloc[-1]}), global + {len(regional_df.columns) - 1} regional series -> {out_path}")
        return len(merged)
    except Exception as exc:
        print(f"[WUI] FAILED — {exc}")
        return 0


def fetch_oni_enso() -> int:
    """NOAA Oceanic Nino Index — seasonal SST anomaly, flagged into El Nino/La Nina/Neutral."""
    out_path = OUT_DIR / "oni_enso.csv"
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        lines = [line.split() for line in response.text.strip().splitlines()]
        header, rows = lines[0], lines[1:]

        df = pd.DataFrame(rows, columns=[h.lower() for h in header])
        df["yr"] = df["yr"].astype(int)
        df["anom"] = df["anom"].astype(float)
        df["month"] = df["seas"].map(SEASON_TO_MONTH)

        def phase(anom: float) -> str:
            if anom >= 0.5:
                return "El Nino"
            if anom <= -0.5:
                return "La Nina"
            return "Neutral"

        result = pd.DataFrame({
            "year": df["yr"],
            "month": df["month"],
            "season": df["seas"],
            "oni_value": df["anom"],
            "enso_phase": df["anom"].map(phase),
        })

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        current = result.iloc[-1]
        print(f"[ONI ENSO] OK — {len(result)} seasonal rows (1950 to {int(current['year'])}), "
              f"current phase: {current['enso_phase']} ({current['oni_value']:+.2f}) -> {out_path}")
        return len(result)
    except Exception as exc:
        print(f"[ONI ENSO] FAILED — {exc}")
        return 0


def fetch_ofac_sanctions() -> int:
    """OFAC Specially Designated Nationals list, including vessel-type entries."""
    out_path = OUT_DIR / "ofac_sanctions.csv"
    url = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.CSV"
    columns = [
        "ent_num", "sdn_name", "sdn_type", "program", "title", "call_sign",
        "vess_type", "tonnage", "grt", "vess_flag", "vess_owner", "remarks",
    ]
    try:
        response = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        response.raise_for_status()
        tmp_path = OUT_DIR / "_sdn_download.csv"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(response.content)

        df = pd.read_csv(tmp_path, header=None, names=columns, encoding="latin-1")
        tmp_path.unlink(missing_ok=True)
        for col in df.columns[1:]:
            df[col] = df[col].astype(str).str.strip().str.strip('"').replace({"-0-": pd.NA})

        df.to_csv(out_path, index=False)
        vessel_count = int((df["sdn_type"] == "vessel").sum())
        program_count = df["program"].nunique()
        print(f"[OFAC Sanctions] OK — {len(df)} SDN entries, {vessel_count} flagged as vessels, "
              f"{program_count} distinct sanctions programs -> {out_path}")
        return len(df)
    except Exception as exc:
        print(f"[OFAC Sanctions] FAILED — {exc}")
        return 0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Phase 3: Risk Indices Ingestion ===")

    gpr_rows = fetch_gpr()
    wui_rows = fetch_wui()
    oni_rows = fetch_oni_enso()
    ofac_rows = fetch_ofac_sanctions()

    print()
    print("[GDELT] SKIPPED — api.gdelt.org has an expired TLS cert and now resolves to a parked/ad "
          "domain; api.gdeltproject.org is unreachable from this network. Documented as future work.")
    print("[ACLED] SKIPPED — data export requires a registered API key we don't have. Documented as future work.")

    print()
    print("=== Summary ===")
    print(f"GPR Index:       {gpr_rows} rows fetched")
    print(f"WUI Index:       {wui_rows} rows fetched")
    print(f"ONI ENSO:        {oni_rows} rows fetched")
    print(f"OFAC Sanctions:  {ofac_rows} rows fetched")


if __name__ == "__main__":
    main()
