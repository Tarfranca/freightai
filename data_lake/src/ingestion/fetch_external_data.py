import csv
import json
import urllib.error
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = BASE_DIR / "raw" / "macro_finance" / "bcb_usd_brl.csv"


def fetch_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def build_urls(start_date: str, end_date: str):
    urls = [
        (
            "CotacaoDolarPeriodo",
            f"https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoDolarPeriodo(dataInicial='{start_date}',dataFinalCotacao='{end_date}')?$format=json",
        ),
        (
            "CotacaoMoedaPeriodo",
            f"https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoMoedaPeriodo(moeda='USD',dataInicial='{start_date}',dataFinalCotacao='{end_date}')?$format=json",
        ),
    ]
    return urls


def main() -> None:
    start_date = "2024-01-01"
    end_date = "2026-01-03"

    records = []
    last_error = None

    for label, url in build_urls(start_date, end_date):
        try:
            payload = fetch_json(url)
            values = payload.get("value", [])
            if values:
                records = values
                print(f"Fetched {len(records)} rows from {label}.")
                break
            print(f"No rows returned for {label}; trying fallback endpoint.")
        except urllib.error.HTTPError as exc:
            last_error = exc
            body = exc.read().decode("utf-8", errors="ignore")
            print(f"Request failed for {label}: {exc.code} - {body[:200]}")
        except Exception as exc:  # pragma: no cover
            last_error = exc
            print(f"Unexpected error for {label}: {exc}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["dataHoraCotacao", "cotacaoCompra", "cotacaoVenda"]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({
                "dataHoraCotacao": row.get("dataHoraCotacao", ""),
                "cotacaoCompra": row.get("cotacaoCompra", ""),
                "cotacaoVenda": row.get("cotacaoVenda", ""),
            })

    print(f"Saved {len(records)} rows to {OUTPUT_PATH}")
    if last_error and not records:
        print("API returned no rows for the requested period. The file was created with headers only.")


if __name__ == "__main__":
    main()
