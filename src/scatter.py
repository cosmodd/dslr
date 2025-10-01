import os.path

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

def scatter_plot(df: pd.DataFrame):
    drop_cols = ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
    features = df.drop(columns=drop_cols, errors='ignore')
    houses = df['Hogwarts House'].unique()

    for f_name, f_values in features.items():
        figure, axes = plt.subplots(3, 4, figsize=(16, 9))
        axes = axes.flatten()
        features_without_comparing = features.drop(columns=f_name, errors='ignore')

        for j, (name, values) in enumerate(features_without_comparing.items()):
            ax = axes[j]

            for house in houses:
                ax.scatter(
                    df.loc[df['Hogwarts House'] == house, f_name],
                    df.loc[df['Hogwarts House'] == house, name],
                    color=house_colors[house],
                    label=house,
                )
                ax.set_xlabel(f_name)
                ax.set_ylabel(name)

        figure.suptitle(f'{f_name} compared to other features')
        figure.legend(houses, loc='upper right')
        figure.tight_layout()

    plt.show()

def main():
    csv_data = pd.read_csv(dataset_path)
    csv_data.set_index('Index', inplace=True)
    scatter_plot(csv_data)

if __name__ == '__main__':
    main()