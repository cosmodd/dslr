import argparse

import numpy as np
import pandas as pd

from operations import count, mean, std, min, max, percentile


def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.select_dtypes(include=[np.number])
    df.set_index('Index', inplace=True)
    return df

def describe(df: pd.DataFrame) -> pd.DataFrame:
    rows = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    result = pd.DataFrame(index=rows)
    df = filter_columns(df)

    for name, series in df.items():
        result[name] = None
        result.at['count', name] = count(series)
        result.at['mean', name] = mean(series)
        result.at['std', name] = std(series)
        result.at['min', name] = min(series)
        result.at['max', name] = max(series)
        result.at['25%', name] = percentile(series, 25)
        result.at['50%', name] = percentile(series, 50)
        result.at['75%', name] = percentile(series, 75)

    result = result.map(lambda v: f'{v:.6f}')
    return result.transpose()

def main():
    parser = argparse.ArgumentParser(
        prog='describe',
        usage='%(prog)s [options]',
        description='Displays information for all numerical features in the CSV file.'
    )
    parser.add_argument('filepath', help='Path to CSV file')
    args = parser.parse_args()

    csv_data = pd.read_csv(args.filepath)
    print(describe(csv_data))

if __name__ == '__main__':
    main()