import numpy as np
import pandas as pd


def count(s: pd.Series) -> int:
    counter = 0

    for v in s:
        if np.isnan(v):
            continue
        counter += 1

    return counter

def sum(s: pd.Series):
    sum_value = s.dtype.type(0)

    for v in s:
        if np.isnan(v):
            continue
        sum_value += v

    return sum_value

def mean(s: pd.Series) -> float:
    return sum(s) / count(s)

def std(s: pd.Series) -> float:
    series_mean = mean(s)
    series_count = count(s)
    squared_deviation_sum = 0.0

    for v in s:
        if np.isnan(v):
            continue
        squared_deviation_sum += (v - series_mean) ** 2

    return np.sqrt(squared_deviation_sum / (count(s) - 1))

def min(s: pd.Series):
    min_value = s.values[0]
    for v in s:
        if np.isnan(v):
            continue

        if v < min_value:
            min_value = v

    return min_value

def max(s: pd.Series):
    max_value = s.values[0]

    for v in s:
        if np.isnan(v):
            continue

        if v > max_value:
            max_value = v

    return max_value

def percentile(s: pd.Series, percentile: float):
    sorted_series = s.dropna().sort_values(ascending=True)
    rank = (percentile / 100) * (len(sorted_series) - 1)
    index = int(rank)
    next_index = index + 1
    lerp_value = rank % 1

    current_value = sorted_series.values[index]
    next_value = sorted_series.values[next_index]
    interpolated_value = current_value + (next_value - current_value) * lerp_value

    return interpolated_value