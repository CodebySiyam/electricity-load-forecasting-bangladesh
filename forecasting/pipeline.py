from __future__ import annotations

import csv
import math
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_EXOGENOUS_COLUMNS = ("generation", "gas", "coal", "liquid_fuel", "load_shedding")


def _parse_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    for fmt in (
        None,
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
    ):
        try:
            if fmt is None:
                return datetime.fromisoformat(normalized)
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse datetime value: {value!r}")


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_time_series_csv(
    csv_path: str,
    datetime_column: str = "datetime",
    demand_column: str = "demand_mw",
    resample_hourly: bool = False,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file is missing a header row.")
        missing = {datetime_column, demand_column} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            timestamp = _parse_datetime(str(row[datetime_column]))
            parsed: Dict[str, object] = {"datetime": timestamp}
            for key, raw_value in row.items():
                if key == datetime_column:
                    continue
                parsed[key] = _safe_float(raw_value)
            records.append(parsed)

    records.sort(key=lambda item: item["datetime"])  # type: ignore[index]
    if resample_hourly:
        records = _resample_hourly_mean(records)
    return records


def _resample_hourly_mean(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[datetime, List[Dict[str, object]]] = {}
    for record in records:
        dt = record["datetime"]  # type: ignore[index]
        assert isinstance(dt, datetime)
        hour_bucket = dt.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(hour_bucket, []).append(record)

    resampled: List[Dict[str, object]] = []
    for bucket_time in sorted(buckets):
        bucket_rows = buckets[bucket_time]
        keys = [k for k in bucket_rows[0].keys() if k != "datetime"]
        averaged: Dict[str, object] = {"datetime": bucket_time}
        for key in keys:
            values = [r[key] for r in bucket_rows if isinstance(r[key], (int, float))]
            averaged[key] = sum(values) / len(values) if values else None
        resampled.append(averaged)
    return resampled


def _population_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def build_supervised_dataset(
    records: Sequence[Dict[str, object]],
    demand_column: str = "demand_mw",
    exogenous_columns: Sequence[str] = DEFAULT_EXOGENOUS_COLUMNS,
    lags: Sequence[int] = (1, 24, 168),
    rolling_windows: Sequence[int] = (24, 168),
    include_fourier_terms: bool = True,
    fourier_order: int = 2,
) -> Tuple[List[Dict[str, float]], List[float], List[datetime]]:
    if not records:
        return [], [], []

    max_lag = max(max(lags), max(rolling_windows))
    available_exogenous = [column for column in exogenous_columns if column in records[0]]
    features: List[Dict[str, float]] = []
    targets: List[float] = []
    timestamps: List[datetime] = []

    for index in range(max_lag, len(records)):
        current = records[index]
        current_dt = current["datetime"]  # type: ignore[index]
        assert isinstance(current_dt, datetime)
        target_value = _safe_float(current.get(demand_column))
        if target_value is None:
            continue

        row_features: Dict[str, float] = {}
        valid = True

        for lag in lags:
            lag_target = _safe_float(records[index - lag].get(demand_column))
            if lag_target is None:
                valid = False
                break
            row_features[f"{demand_column}_lag_{lag}"] = lag_target

        if not valid:
            continue

        for column in available_exogenous:
            for lag in lags:
                lag_value = _safe_float(records[index - lag].get(column))
                if lag_value is None:
                    valid = False
                    break
                row_features[f"{column}_lag_{lag}"] = lag_value
            if not valid:
                break

        if not valid:
            continue

        for window in rolling_windows:
            history = [
                _safe_float(records[i].get(demand_column))
                for i in range(index - window, index)
            ]
            if any(v is None for v in history):
                valid = False
                break
            valid_history = [float(v) for v in history if v is not None]
            row_features[f"{demand_column}_rolling_mean_{window}"] = sum(valid_history) / len(valid_history)
            row_features[f"{demand_column}_rolling_std_{window}"] = _population_std(valid_history)

        if not valid:
            continue

        row_features["hour"] = float(current_dt.hour)
        row_features["day"] = float(current_dt.day)
        row_features["month"] = float(current_dt.month)
        row_features["weekday"] = float(current_dt.weekday())

        if include_fourier_terms:
            daily_angle = 2.0 * math.pi * (current_dt.hour / 24.0)
            weekly_angle = 2.0 * math.pi * ((current_dt.weekday() * 24 + current_dt.hour) / (7.0 * 24.0))
            for k in range(1, fourier_order + 1):
                row_features[f"daily_sin_{k}"] = math.sin(k * daily_angle)
                row_features[f"daily_cos_{k}"] = math.cos(k * daily_angle)
                row_features[f"weekly_sin_{k}"] = math.sin(k * weekly_angle)
                row_features[f"weekly_cos_{k}"] = math.cos(k * weekly_angle)

        features.append(row_features)
        targets.append(target_value)
        timestamps.append(current_dt)

    return features, targets, timestamps


def time_based_split(
    features: Sequence[Dict[str, float]],
    targets: Sequence[float],
    timestamps: Sequence[datetime],
    test_fraction: float = 0.2,
) -> Tuple[List[Dict[str, float]], List[float], List[datetime], List[Dict[str, float]], List[float], List[datetime]]:
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be between 0 and 1.")
    if not (len(features) == len(targets) == len(timestamps)):
        raise ValueError("features, targets, and timestamps must have equal length.")
    if len(features) < 2:
        raise ValueError("At least two samples are required for a train/test split.")

    split_index = int(len(features) * (1.0 - test_fraction))
    split_index = max(1, min(split_index, len(features) - 1))

    return (
        list(features[:split_index]),
        list(targets[:split_index]),
        list(timestamps[:split_index]),
        list(features[split_index:]),
        list(targets[split_index:]),
        list(timestamps[split_index:]),
    )


def persistence_forecast_walk_forward(train_targets: Sequence[float], test_targets: Sequence[float]) -> List[float]:
    if not train_targets:
        raise ValueError("train_targets must contain at least one value.")

    history = [float(v) for v in train_targets]
    predictions: List[float] = []
    for actual in test_targets:
        prediction = history[-1]
        predictions.append(prediction)
        history.append(float(actual))
    return predictions


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted lengths must match.")
    if not actual:
        raise ValueError("actual cannot be empty.")
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted lengths must match.")
    if not actual:
        raise ValueError("actual cannot be empty.")
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def save_forecast_plot_svg(
    timestamps: Sequence[datetime],
    actual: Sequence[float],
    predicted: Sequence[float],
    output_path: str,
) -> None:
    if not (len(timestamps) == len(actual) == len(predicted)):
        raise ValueError("timestamps, actual, and predicted lengths must match.")
    if not timestamps:
        raise ValueError("Cannot create a plot without values.")

    width, height, margin = 1000, 400, 50
    ymin = min(min(actual), min(predicted))
    ymax = max(max(actual), max(predicted))
    if ymin == ymax:
        ymax = ymin + 1.0

    def scale_x(i: int) -> float:
        if len(timestamps) == 1:
            return margin
        return margin + (width - 2 * margin) * (i / (len(timestamps) - 1))

    def scale_y(v: float) -> float:
        return height - margin - ((v - ymin) / (ymax - ymin)) * (height - 2 * margin)

    actual_points = " ".join(f"{scale_x(i):.2f},{scale_y(v):.2f}" for i, v in enumerate(actual))
    pred_points = " ".join(f"{scale_x(i):.2f},{scale_y(v):.2f}" for i, v in enumerate(predicted))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(
            f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>
<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{actual_points}"/>
<polyline fill="none" stroke="#d62728" stroke-width="2" points="{pred_points}"/>
<text x="{margin}" y="{margin-12}" fill="#1f77b4">Actual</text>
<text x="{margin+70}" y="{margin-12}" fill="#d62728">Forecast</text>
</svg>
"""
        )


def _write_forecast_csv(
    output_path: str,
    timestamps: Sequence[datetime],
    actual: Sequence[float],
    predicted: Sequence[float],
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["datetime", "actual_demand_mw", "forecast_demand_mw"])
        for dt, a, p in zip(timestamps, actual, predicted):
            writer.writerow([dt.isoformat(sep=" "), f"{a:.6f}", f"{p:.6f}"])


def run_forecasting_pipeline(
    csv_path: str,
    output_dir: str,
    datetime_column: str = "datetime",
    demand_column: str = "demand_mw",
    test_fraction: float = 0.2,
    resample_hourly: bool = False,
) -> Dict[str, object]:
    records = load_time_series_csv(
        csv_path=csv_path,
        datetime_column=datetime_column,
        demand_column=demand_column,
        resample_hourly=resample_hourly,
    )
    features, targets, timestamps = build_supervised_dataset(records, demand_column=demand_column)
    _, train_y, _, _, test_y, test_ts = time_based_split(features, targets, timestamps, test_fraction=test_fraction)
    predictions = persistence_forecast_walk_forward(train_y, test_y)

    plot_path = os.path.join(output_dir, "forecast_vs_actual.svg")
    forecast_csv_path = os.path.join(output_dir, "forecast_results.csv")
    save_forecast_plot_svg(test_ts, test_y, predictions, plot_path)
    _write_forecast_csv(forecast_csv_path, test_ts, test_y, predictions)

    return {
        "model": "persistence_baseline",
        "samples_total": len(targets),
        "samples_train": len(train_y),
        "samples_test": len(test_y),
        "mae": mae(test_y, predictions),
        "rmse": rmse(test_y, predictions),
        "forecast_plot_path": plot_path,
        "forecast_csv_path": forecast_csv_path,
    }


def walk_forward_splits(length: int, initial_train_size: int, step_size: int = 1) -> Iterable[Tuple[range, range]]:
    if length <= 0:
        raise ValueError("length must be > 0")
    if initial_train_size <= 0 or initial_train_size >= length:
        raise ValueError("initial_train_size must be between 1 and length-1.")
    if step_size <= 0:
        raise ValueError("step_size must be > 0")

    train_end = initial_train_size
    while train_end < length:
        test_end = min(length, train_end + step_size)
        yield range(0, train_end), range(train_end, test_end)
        train_end = test_end
