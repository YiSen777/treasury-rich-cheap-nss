# U.S. Treasury Rich-Cheap Strategy with NSS Curve Fitting

This project implements a fixed-income relative-value research pipeline for the U.S. Treasury curve. It downloads Treasury yield data from FRED, fits a Nelson-Siegel-Svensson (NSS) fair-value curve, identifies rich and cheap maturity buckets from residual z-scores, and backtests a duration-adjusted long-cheap / short-rich strategy.

## Resume Summary

Built a Python-based Treasury relative-value strategy that calibrates Nelson-Siegel-Svensson yield curves, validates yield-curve factor structure with KMO and PCA, generates residual z-score trading signals, and evaluates strategy performance against passive Treasury benchmarks with transaction-cost assumptions.

## Project Highlights

- Calibrates daily NSS yield curves across 3M to 30Y U.S. Treasury maturities.
- Uses residual z-scores to identify maturities trading rich or cheap versus model fair value.
- Constructs duration-adjusted long/short positions to reduce rate-risk concentration.
- Backtests net returns with turnover-based transaction costs.
- Benchmarks the strategy against long 10Y, long 20Y, and equal-weight Treasury curve exposures.
- Includes KMO and PCA analysis to support level, slope, and curvature interpretation.

## Methodology

1. Download daily Treasury par yield data from FRED.
2. Convert maturity labels into year terms for NSS fitting.
3. Calibrate the NSS curve each day using bounded numerical optimization.
4. Compute market-minus-model residuals for each maturity bucket.
5. Normalize residuals into rolling z-scores.
6. Go long the cheapest maturities and short the richest maturities.
7. Scale weights by inverse duration and target gross leverage.
8. Approximate bond returns from yield changes and duration.
9. Compare the strategy with passive Treasury benchmarks.

## Repository Structure

```text
treasury-rich-cheap-nss/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── treasury_rich_cheap_nss.py
└── outputs/
```

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/treasury_rich_cheap_nss.py
```

The script fetches data directly from FRED and displays the performance summary and plots.

## Notes

This project is designed as a research prototype, not a production trading system. The duration values are simple proxies, and a production implementation should replace them with instrument-level DV01, execution constraints, financing assumptions, and a more robust data pipeline.

