import os

import pandas as pd
from matplotlib import pyplot as plt

parent_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(parent_path, '../datasets/dataset_train.csv')

house_colors = {
    'Ravenclaw': '#222f5b',
    'Slytherin': '#2a623d',
    'Gryffindor': '#ae0001',
    'Hufflepuff': '#f0c75e'
}

def pair_plot(df: pd.DataFrame):
    drop_cols = ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
    features = df.drop(columns=drop_cols, errors='ignore')
    columns = features.columns
    houses = df['Hogwarts House'].unique()

    figure, axes = plt.subplots(len(columns), len(columns), figsize=(16, 9))

    for y, y_col in enumerate(columns):

        for x, x_col in enumerate(columns):
            ax = axes[y, x]
            ax.set_xticks([])
            ax.set_yticks([])

            # Labels
            if x == 0:
                ax.set_ylabel(y_col, rotation=75, loc='top')

            if y == len(columns) - 1:
                ax.set_xlabel(x_col, rotation=15, loc='right')

            for house in houses:
                x_values = df.loc[df['Hogwarts House'] == house, x_col]
                y_values = df.loc[df['Hogwarts House'] == house, y_col]

                # Scatter plot
                if x != y:
                    ax.scatter(
                        x_values,
                        y_values,
                        color=house_colors[house],
                        s=2,
                    )

                # Histogram plot
                else:
                    ax.hist(
                        x_values,
                        color=house_colors[house],
                        alpha=0.75,
                        bins=50,
                    )

    figure.suptitle('Pair plot')
    figure.legend(houses, loc='upper right')
    figure.tight_layout()
    plt.show()

def main():
    csv_data = pd.read_csv(dataset_path)
    csv_data.set_index('Index', inplace=True)
    pair_plot(csv_data)

if __name__ == '__main__':
    main()