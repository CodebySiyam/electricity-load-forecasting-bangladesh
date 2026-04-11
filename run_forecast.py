#!/usr/bin/env python3
import argparse
import json

from forecasting import run_forecasting_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe electricity demand forecasting baseline.")
    parser.add_argument("--csv", required=True, help="Path to input CSV containing datetime and demand_mw columns.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to write forecast outputs.")
    parser.add_argument("--datetime-column", default="datetime", help="Datetime column name in the CSV.")
    parser.add_argument("--demand-column", default="demand_mw", help="Demand column name in the CSV.")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Chronological test fraction (0,1).")
    parser.add_argument(
        "--resample-hourly",
        action="store_true",
        help="Resample data to hourly means before feature engineering.",
    )
    args = parser.parse_args()

    result = run_forecasting_pipeline(
        csv_path=args.csv,
        output_dir=args.output_dir,
        datetime_column=args.datetime_column,
        demand_column=args.demand_column,
        test_fraction=args.test_fraction,
        resample_hourly=args.resample_hourly,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
