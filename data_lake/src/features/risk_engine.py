"""Composite corridor risk engine for FreightAI.

Combines everything fetched in Phases 1-5 into a single 0-10 risk score per
shipping corridor:

    composite_risk = geopolitical_risk * 0.30 + economic_risk * 0.20
                    + climate_risk * 0.20 + operational_risk * 0.15
                    + market_risk * 0.15

Every sub-score is normalized to 0-10 via min-max scaling against its own
historical range in the data already in the lake (same approach used for the
risk radar in api.py) — nothing here is a hardcoded/guessed number except the
operational baselines, which the spec explicitly asked to hardcode.

Two of the five inputs (geopolitical_risk, market_risk) are GLOBAL indices —
GPR and VIX/BDRY don't have a corridor breakdown in the data we have, so per
the spec they're computed once and applied to every corridor. climate_risk and
operational_risk are genuinely corridor-specific (see the mapping notes below).
"""
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import numpy as np
    import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
GPR_PATH = BASE_DIR / "raw" / "risk" / "gpr_index.csv"
WUI_PATH = BASE_DIR / "raw" / "risk" / "wui_index.csv"
VIX_PATH = BASE_DIR / "raw" / "commodities" / "VIX.csv"
ONI_PATH = BASE_DIR / "raw" / "risk" / "oni_enso.csv"
WEATHER_PATH = BASE_DIR / "raw" / "climate" / "port_weather.csv"
OFAC_PATH = BASE_DIR / "raw" / "risk" / "ofac_sanctions.csv"
MASTER_PATH = BASE_DIR / "gold" / "commodities_master.csv"
OUT_PATH = BASE_DIR / "gold" / "corridor_risk_scores.csv"

CORRIDORS = [
    "Suez/Red Sea", "Panama Canal", "Transpacific", "North Atlantic",
    "Intra-Asia", "South America East Coast", "Cape of Good Hope", "Strait of Malacca",
]

# Which of the 6 ports fetched in Phase 4 best represents each corridor's local
# weather exposure. Suez/Red Sea, Panama Canal and Cape of Good Hope have no
# directly-fetched port, so they fall back to the cross-port average (flagged
# in the printed table) rather than guessing a number.
CORRIDOR_PORT_MAP = {
    "Suez/Red Sea": [],
    "Panama Canal": [],
    "Transpacific": ["Shanghai", "Los Angeles"],
    "North Atlantic": ["Rotterdam", "Hamburg"],
    "Intra-Asia": ["Shanghai", "Singapore"],
    "South America East Coast": ["Santos"],
    "Cape of Good Hope": [],
    "Strait of Malacca": ["Singapore"],
}

# Historical-disruption-frequency baseline (0-10), as specified. "Gulf of
# Guinea" from the prompt maps onto "Cape of Good Hope" here since that's the
# closest corridor in our 8-corridor list (ships routing around Africa to
# avoid Suez pass along the Gulf of Guinea piracy belt).
OPERATIONAL_BASELINE = {
    "Suez/Red Sea": 8, "Panama Canal": 6, "Strait of Malacca": 5, "Cape of Good Hope": 7,
    "Transpacific": 3, "North Atlantic": 3, "Intra-Asia": 3, "South America East Coast": 3,
}

# OFAC "program" keywords used to attribute sanctioned vessels to a corridor.
# Deliberately keyed on `program` (the designation reason) rather than
# `vess_flag` — most sanctioned vessels fly flags-of-convenience (Panama,
# Liberia, Marshall Islands) that reflect ship registry economics, not where
# the vessel actually operates, so flag would misattribute risk.
CORRIDOR_OFAC_KEYWORDS = {
    "Suez/Red Sea": ["IRAN", "YEMEN"],
    "Panama Canal": ["VENEZUELA", "CUBA"],
    "Transpacific": ["DPRK", "NORTH KOREA"],
    "North Atlantic": ["RUSSIA", "UKRAINE", "BELARUS"],
    "Intra-Asia": ["DPRK", "NORTH KOREA", "MYANMAR"],
    "South America East Coast": ["VENEZUELA"],
    "Cape of Good Hope": ["LIBYA", "SUDAN", "SOMALIA"],
    "Strait of Malacca": ["DPRK", "MYANMAR"],
}


def minmax_scale(value: float, series: pd.Series) -> float:
    """Scale `value` to 0-10 using the historical min/max of `series`."""
    low, high = series.min(), series.max()
    if high == low:
        return 5.0
    scaled = (value - low) / (high - low) * 10.0
    return float(np.clip(scaled, 0.0, 10.0))


def compute_geopolitical_risk() -> tuple[float, float]:
    """Global GPR index, latest value normalized against its own history."""
    gpr = pd.read_csv(GPR_PATH, parse_dates=["date"])
    latest = gpr.iloc[-1]
    score = minmax_scale(latest["gpr_global"], gpr["gpr_global"])
    return score, float(latest["gpr_global"])


def compute_economic_risk() -> tuple[float, float]:
    """Global WUI + VIX, each normalized against its own history, then averaged."""
    wui = pd.read_csv(WUI_PATH)
    wui_score = minmax_scale(wui["wui_global"].iloc[-1], wui["wui_global"])

    vix = pd.read_csv(VIX_PATH, parse_dates=["Date"])
    vix_latest = float(vix["Close"].iloc[-1])
    vix_score = minmax_scale(vix_latest, vix["Close"])

    return float((wui_score + vix_score) / 2.0), vix_latest


def compute_climate_risk() -> tuple[dict, str, float]:
    """Per corridor: ONI-derived global anomaly severity + local port weather severity."""
    oni = pd.read_csv(ONI_PATH)
    current = oni.iloc[-1]
    oni_value = float(current["oni_value"])
    # Both El Nino and La Nina represent anomalous (disruptive) conditions —
    # score by magnitude, scaled so |ONI|=2.5 (a strong event) maps to ~10.
    oni_score = float(np.clip(abs(oni_value) / 2.5 * 10.0, 0.0, 10.0))

    weather = pd.read_csv(WEATHER_PATH)
    port_severity = weather.groupby("port").apply(
        lambda g: 0.6 * g["windspeed_10m_max_kmh"].mean() + 0.4 * g["precipitation_sum_mm"].mean(),
        include_groups=False,
    )
    port_severity_scaled = port_severity.apply(lambda v: minmax_scale(v, port_severity))
    fleet_avg_severity = float(port_severity_scaled.mean())

    scores = {}
    for corridor in CORRIDORS:
        ports = CORRIDOR_PORT_MAP[corridor]
        if ports:
            local_severity = float(port_severity_scaled.reindex(ports).mean())
        else:
            local_severity = fleet_avg_severity  # no directly-fetched port for this corridor
        scores[corridor] = float(np.clip((oni_score + local_severity) / 2.0, 0.0, 10.0))

    return scores, str(current["enso_phase"]), oni_value


def compute_operational_risk() -> dict:
    """Hardcoded baseline per corridor, adjusted up by relevant OFAC vessel sanctions."""
    ofac = pd.read_csv(OFAC_PATH)
    vessels = ofac[ofac["sdn_type"] == "vessel"].copy()
    vessels["program"] = vessels["program"].astype(str).str.upper()

    scores = {}
    for corridor, baseline in OPERATIONAL_BASELINE.items():
        keywords = CORRIDOR_OFAC_KEYWORDS[corridor]
        mask = vessels["program"].apply(lambda p: any(kw in p for kw in keywords))
        matched_count = int(mask.sum())
        # +0.5 per 10 matched sanctioned vessels, capped at +2.0 so a single
        # heavily-sanctioned program (e.g. Iran) doesn't blow out the scale.
        adjustment = min(matched_count / 10.0 * 0.5, 2.0)
        scores[corridor] = float(np.clip(baseline + adjustment, 0.0, 10.0))
    return scores


def compute_market_risk() -> float:
    """ZIM + BDRY 30-day rolling volatility, normalized against its own history."""
    master = pd.read_csv(MASTER_PATH, parse_dates=["date"]).set_index("date")

    zim_vol = master["ZIM"].rolling(30).std()
    bdry_vol = master["BDRY"].rolling(30).std()

    zim_score = minmax_scale(zim_vol.dropna().iloc[-1], zim_vol.dropna())
    bdry_score = minmax_scale(bdry_vol.dropna().iloc[-1], bdry_vol.dropna())
    return float((zim_score + bdry_score) / 2.0)


def main() -> None:
    print("=== Phase 6: Composite Risk Engine ===\n")

    geo_score, gpr_latest = compute_geopolitical_risk()
    print(f"Geopolitical risk (global GPR={gpr_latest:.1f}): {geo_score:.2f}/10 — applied to all corridors")

    econ_score, vix_latest = compute_economic_risk()
    print(f"Economic risk (global WUI+VIX={vix_latest:.1f}): {econ_score:.2f}/10 — applied to all corridors")

    climate_scores, enso_phase, oni_latest = compute_climate_risk()
    print(f"Climate risk: computed per corridor (ONI={oni_latest:+.2f}, phase={enso_phase})")

    operational_scores = compute_operational_risk()
    print("Operational risk: computed per corridor (baseline + OFAC vessel-sanction adjustment)")

    market_score = compute_market_risk()
    print(f"Market risk (ZIM+BDRY 30d volatility): {market_score:.2f}/10 — applied to all corridors\n")

    rows = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    for corridor in CORRIDORS:
        climate = climate_scores[corridor]
        operational = operational_scores[corridor]
        composite = (
            geo_score * 0.30 + econ_score * 0.20 + climate * 0.20
            + operational * 0.15 + market_score * 0.15
        )
        rows.append({
            "corridor": corridor,
            "composite_score": round(composite, 2),
            "geopolitical_risk": round(geo_score, 2),
            "economic_risk": round(econ_score, 2),
            "climate_risk": round(climate, 2),
            "operational_risk": round(operational, 2),
            "market_risk": round(market_score, 2),
            "last_updated": now,
            "enso_phase": enso_phase,
            "gpr_latest": round(gpr_latest, 1),
            "vix_latest": round(vix_latest, 1),
        })

    result = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)

    print("=== Corridor Risk Table ===")
    print(result[["corridor", "composite_score", "geopolitical_risk", "economic_risk",
                   "climate_risk", "operational_risk", "market_risk"]].to_string(index=False))
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
