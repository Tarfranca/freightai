"""Cleans the OpenFlights airports database and the World Port Index (Pub150)
into gold/airports_clean.csv and gold/ports_clean.csv for the maritime/air
prediction endpoints in api.py.
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
AIRPORTS_RAW = BASE_DIR / "raw" / "reference" / "airports.csv"
PORTS_RAW = BASE_DIR / "raw" / "ports" / "UpdatedPub150 (1).csv"
AIRPORTS_OUT = BASE_DIR / "gold" / "airports_clean.csv"
PORTS_OUT = BASE_DIR / "gold" / "ports_clean.csv"

AIRPORT_COLUMNS = [
    "airport_id", "name", "city", "country", "iata", "icao",
    "lat", "lon", "altitude", "timezone", "dst", "tz", "type", "source",
]


def build_airports():
    df = pd.read_csv(AIRPORTS_RAW, header=None, names=AIRPORT_COLUMNS, na_values=["\\N"])
    df["iata"] = df["iata"].astype(str).str.strip()
    df = df[df["iata"].notna() & (df["iata"] != "") & (df["iata"] != "\\N") & (df["iata"] != "nan")]
    df = df.dropna(subset=["lat", "lon"])
    df.to_csv(AIRPORTS_OUT, index=False)
    print(f"airports_clean.csv: {len(df)} rows -> {AIRPORTS_OUT}")


def build_ports():
    df = pd.read_csv(PORTS_RAW, low_memory=False)
    cleaned = pd.DataFrame({
        "port_name": df["Main Port Name"],
        "country": df["Country Code"],
        "latitude": df["Latitude"],
        "longitude": df["Longitude"],
        "port_size": df["Harbor Size"],
        "water_body": df["World Water Body"],
        "shelter": df["Shelter Afforded"],
        "max_vessel_size": df["Maximum Vessel Length (m)"],
    })
    cleaned = cleaned.dropna(subset=["port_name", "latitude", "longitude"])
    cleaned = cleaned[cleaned["port_name"].astype(str).str.strip() != ""]
    cleaned.to_csv(PORTS_OUT, index=False)
    print(f"ports_clean.csv: {len(cleaned)} rows -> {PORTS_OUT}")


if __name__ == "__main__":
    build_airports()
    build_ports()
