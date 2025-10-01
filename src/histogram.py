from argparse import ArgumentParser

import pandas as pd
from matplotlib import pyplot as plt


def plot_histogram(df: pd.DataFrame):
    drop_cols = ["Index", "Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"]
    features = df.drop(columns=drop_cols, errors="ignore").columns
    houses = df["Hogwarts House"].unique()

    house_colors = {
        'Ravenclaw': '#222f5b',
        'Slytherin': '#2a623d',
        'Gryffindor': '#ae0001',
        'Hufflepuff': '#f0c75e'
    }

    fig, axes = plt.subplots(4, 4, figsize=(16, 9))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        for house in houses:
            df.loc[df["Hogwarts House"] == house, feature].plot.hist(
                ax=ax,
                bins=50,
                alpha=0.75,
                color=house_colors[house],
                label=house
            )
        ax.set_title(feature)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.legend(houses, loc="upper right")
    plt.tight_layout()
    plt.show()

def main():
    parser = ArgumentParser(
        prog='histogram.py',
        description='Plot histogram.'
    )
    parser.add_argument('filepath', help='Path to CSV file')
    args = parser.parse_args()

    csv_data = pd.read_csv(args.filepath)
    plot_histogram(csv_data)

if __name__ == '__main__':
    main()