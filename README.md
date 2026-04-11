# multivariate-energy-demand-forecasting

Leakage-safe multivariate time-series forecasting pipeline for electricity demand (`demand_mw`) with:

- Chronological datetime handling (+ optional hourly resampling)
- Lag features (`t-1`, `t-24`, `t-168`) for demand and exogenous signals
- Rolling mean/std windows (`24h`, `7d`) using only past observations
- Calendar features (`hour`, `day`, `month`, `weekday`) and optional Fourier seasonality terms
- Time-based train/test split (no shuffle)
- Walk-forward persistence baseline
- RMSE/MAE evaluation
- Forecast-vs-actual plot output (`SVG`)

## Expected input columns

- `datetime`
- `demand_mw`
- Exogenous columns (if present): `generation`, `gas`, `coal`, `liquid_fuel`, `load_shedding`

## Run baseline forecast

```bash
python run_forecast.py --csv /absolute/path/to/data.csv --output-dir /absolute/path/to/output
```

Outputs:
- `/absolute/path/to/output/forecast_results.csv`
- `/absolute/path/to/output/forecast_vs_actual.svg`

## Run tests

```bash
python -m unittest discover -s tests -q
```
