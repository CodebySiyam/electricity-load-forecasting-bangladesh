import csv
import os
import tempfile
import unittest
from datetime import datetime, timedelta

from forecasting.pipeline import (
    build_supervised_dataset,
    load_time_series_csv,
    persistence_forecast_walk_forward,
    rmse,
    run_forecasting_pipeline,
    time_based_split,
    walk_forward_splits,
)


def _synthetic_records(count: int = 220):
    start = datetime(2026, 1, 1, 0, 0, 0)
    records = []
    for i in range(count):
        records.append(
            {
                "datetime": start + timedelta(hours=i),
                "demand_mw": float(1000 + i),
                "generation": float(900 + i * 0.5),
                "gas": float(300 + i * 0.2),
                "coal": float(200 + i * 0.1),
                "liquid_fuel": float(80 + i * 0.05),
                "load_shedding": float((i % 3) * 2),
            }
        )
    return records


class PipelineTests(unittest.TestCase):
    def test_feature_engineering_uses_only_past_values(self):
        records = _synthetic_records()
        features, targets, timestamps = build_supervised_dataset(records)

        self.assertTrue(features)
        row = features[0]
        self.assertEqual(targets[0], records[168]["demand_mw"])
        self.assertEqual(timestamps[0], records[168]["datetime"])
        self.assertEqual(row["demand_mw_lag_1"], records[167]["demand_mw"])
        self.assertEqual(row["demand_mw_lag_24"], records[144]["demand_mw"])
        self.assertEqual(row["demand_mw_lag_168"], records[0]["demand_mw"])

        expected_mean_24 = sum(records[i]["demand_mw"] for i in range(144, 168)) / 24
        self.assertAlmostEqual(row["demand_mw_rolling_mean_24"], expected_mean_24)

    def test_time_split_preserves_chronological_order(self):
        records = _synthetic_records()
        features, targets, timestamps = build_supervised_dataset(records)
        _, train_y, train_ts, _, test_y, test_ts = time_based_split(features, targets, timestamps, test_fraction=0.25)

        self.assertTrue(train_y)
        self.assertTrue(test_y)
        self.assertLess(train_ts[-1], test_ts[0])

    def test_walk_forward_baseline_and_outputs(self):
        records = _synthetic_records()
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "data.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["datetime", "demand_mw", "generation", "gas", "coal", "liquid_fuel", "load_shedding"]
                )
                shuffled = sorted(records, key=lambda r: r["datetime"], reverse=True)
                for row in shuffled:
                    writer.writerow(
                        [
                            row["datetime"].isoformat(sep=" "),
                            row["demand_mw"],
                            row["generation"],
                            row["gas"],
                            row["coal"],
                            row["liquid_fuel"],
                            row["load_shedding"],
                        ]
                    )

            loaded = load_time_series_csv(csv_path)
            self.assertLess(loaded[0]["datetime"], loaded[-1]["datetime"])

            result = run_forecasting_pipeline(csv_path=csv_path, output_dir=tmp_dir)
            self.assertEqual(result["model"], "persistence_baseline")
            self.assertGreater(result["samples_test"], 0)
            self.assertTrue(os.path.exists(result["forecast_plot_path"]))
            self.assertTrue(os.path.exists(result["forecast_csv_path"]))

            forecast = persistence_forecast_walk_forward([1.0, 2.0, 3.0], [4.0, 5.0])
            self.assertEqual(forecast, [3.0, 4.0])
            self.assertAlmostEqual(rmse([4.0, 5.0], forecast), ((1.0 + 1.0) / 2.0) ** 0.5)

    def test_walk_forward_splits_progression(self):
        splits = list(walk_forward_splits(length=10, initial_train_size=4, step_size=2))
        as_ranges = [((s.start, s.stop), (t.start, t.stop)) for s, t in splits]
        self.assertEqual(as_ranges, [((0, 4), (4, 6)), ((0, 6), (6, 8)), ((0, 8), (8, 10))])


if __name__ == "__main__":
    unittest.main()
