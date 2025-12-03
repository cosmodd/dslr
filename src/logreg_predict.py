import pickle
import sys
import pandas as pd
import numpy as np
from logreg_train import sigmoid

def load_data(dataset_path, model_path):
    df = pd.read_csv(dataset_path)
    df = df.set_index('Index')
    with open(model_path, 'rb') as f:
        save_data = pickle.load(f)
    models = save_data['models']
    X_mean = save_data['X_mean']
    X_std = save_data['X_std']
    return df, models, X_mean, X_std

def preprocess_data(df):
    best_hand_mapping = {
        "Left": 0,
        "Right": 1 
    }
    df['Best Hand'] = df['Best Hand'].map(best_hand_mapping)
    df = df.drop(columns=['First Name', 'Last Name', 'Birthday', 'Hogwarts House'], errors='ignore')
    return df

def scale_features(X, X_mean, X_std):
    X_scaled = (X - X_mean) / X_std
    return X_scaled

def predict(X_scaled, models):
    m = X_scaled.shape[0]
    num_classes = len(models)
    probabilities = np.zeros((m, num_classes)) # 2D array (samples x classes)
    for i, (W, b) in enumerate(models):
        Z = np.dot(X_scaled, W) + b
        A = sigmoid(Z)
        probabilities[:, i] = A
    predictions = np.argmax(probabilities, axis=1)
    return predictions

def main():
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py <model_path> <dataset_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    dataset_path = sys.argv[2]
    df, models, X_mean, X_std = load_data(dataset_path, model_path)
    preprocessed_df = preprocess_data(df)
    X = preprocessed_df.values
    X_scaled = scale_features(X, X_mean, X_std)
    predictions = predict(X_scaled, models)
    index_list = preprocessed_df.index.tolist()
    with open('logreg_predictions.txt', 'w') as f:
        f.write("Index,Hogwarts House\n")
        for idx, pred in zip(index_list, predictions):
            f.write(f"{idx},{['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'][pred]}\n")
    print("Predictions saved to logreg_predictions.txt")

if __name__ == "__main__":
    main()