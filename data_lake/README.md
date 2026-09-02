# Data Lake structure

This folder organizes the raw CSVs into a simple lake architecture so they can be consumed by Spark, SQL, or a warehouse later.

## Layers
- raw/: immutable source files organized by business domain.
- bronze/: normalized copies with cleaned filenames but no transformation logic.
- silver/: cleaned and standardized tables ready for business rules and QA.
- gold/: curated analytical tables for dashboards and reporting.
- metadata/: dataset catalog and schema inventory.
- warehouse/: SQLite warehouse schema and curated fact tables.

## Normalized domains
- energy
- commodity
- ports
- trade
- supply_chain
- macro_finance
- market

## Current datasets
The source folder contains 11 CSV files covering:
- oil and fuel prices
- commodity price series
- port activity and port publication data
- import/export trade data
- logistics and supply chain risk information
- macro-financial indicators
- market trends and market risk data

## Build commands
From the project root:

python data_lake/build_datalake.py
python data_lake/build_warehouse.py

The warehouse builder standardizes the source headers, writes the silver layer, and loads a star-style SQLite warehouse schema for analysis.
