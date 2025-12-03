import sys
import pandas as pd
import numpy as np
import pickle

def sigmoid(z):
    # Squash to 0-1 prob
    return 1 / (1 + np.exp(-z))

def load_data(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df.set_index('Index')
    print(f"Data loaded from {dataset_path}. Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
    print(f"----------------- Data Preview ------------------\n{df.head()}\n-------------------------------------------------")
    return df

def preprocess_data(df):
    # Map houses, hands & drop useless cols
    house_mapping = {
        "Gryffindor": 0,
        "Hufflepuff": 1,
        "Ravenclaw": 2,
        "Slytherin": 3
    }
    df['Hogwarts House'] = df['Hogwarts House'].map(house_mapping)
    best_hand_mapping = {
        "Left": 0,
        "Right": 1
    }
    df['Best Hand'] = df['Best Hand'].map(best_hand_mapping)
    df = df.drop(columns=['First Name', 'Last Name', 'Birthday'])
    # If any NaN values, replace NaN with mean of the column across all samples
    # df = df.dropna()
    df = df.fillna(df.mean())
    print(f"----------- Preprocessed Data Preview -----------\n{df.head()}\n-------------------------------------------------")
    return df

def scale_features(X):
    # Standardization : https://en.wikipedia.org/wiki/Feature_scaling#Standardization_(Z-score_Normalization)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / (X_std)
    return X_scaled, X_mean, X_std

def binary_model(X, Y, learning_rate, epochs):
    """
    Gradient descent to minimize log loss L = - (1/N) * Σ [y * log(a) + (1 - y) * log(1 - a)]
    https://community.deeplearning.ai/t/why-mse-is-not-a-good-loss-function-for-logistic-regression/255547
    """
    # Setup: samples m, features n, weights W, bias b
    m = X.shape[0]
    n = X.shape[1]
    W = np.zeros(n)
    b = 0
    for epoch in range(epochs):
        # Linear score (Z = W.X + b) and sigmoid for probs
        Z = np.dot(X, W) + b
        A = sigmoid(Z)
        # How much L increases w.r.t W and b ? Partial derivatives: Chain rule to find : dW = dL/dA * dA/dZ * dZ/dW | db = dL/dA * dA/dZ * dZ/db
        dW = (1/m) * np.dot(X.T, (A - Y)) # Transpose data matrix to match dimensions with error vector
        db = (1/m) * np.sum(A - Y)
        # Step opposite gradient
        W -= learning_rate * dW
        b -= learning_rate * db
        if epoch % 100 == 0:
            loss = - (1/m) * np.sum(Y * np.log(A + 1e-15) + (1 - Y) * np.log(1 - A + 1e-15))
            print(f"Binary Model - Epoch {epoch}, Loss: {loss:.4f}")
    return W, b

def train_ovr(X, Y, learning_rate, epochs):
    # Iterate over each class and train a binary model
    classes = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    models = []
    for k in range(len(classes)):
        print(f"\n--- Training ovr model for class {k} (e.g., {classes[k]}) ---")
        y_bin = (Y == k).astype(float)
        W, b = binary_model(X, y_bin, learning_rate, epochs)
        models.append((W, b))
    return models

def main():
    if len(sys.argv) != 2 or not sys.argv[1].endswith('.csv'):
        print("Usage: python logreg_train.py <dataset_path>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    df = load_data(dataset_path)
    preprocessed_df = preprocess_data(df)
    # Split features and labels
    X = preprocessed_df.drop(columns=['Hogwarts House']).values
    Y = preprocessed_df['Hogwarts House'].values
    print(f"Train X shape: {X.shape}, Train Y shape: {Y.shape}")
    # Scale features, train & save models
    X_scaled, X_mean, X_std = scale_features(X)
    models = train_ovr(X_scaled, Y, learning_rate=0.1, epochs=100000)
    save_data = {
        'models': models,
        'X_mean': X_mean,
        'X_std': X_std
    }
    # Test precision on training set
    m = X_scaled.shape[0]
    num_classes = len(models)
    probabilities = np.zeros((m, num_classes))
    for i, (W, b) in enumerate(models):
        Z = np.dot(X_scaled, W) + b
        A = sigmoid(Z)
        probabilities[:, i] = A
    predictions = np.argmax(probabilities, axis=1)
    accuracy = np.mean(predictions == Y)
    print(f"\nTraining accuracy: {accuracy * 100:.2f}%")
    with open('logreg_models.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    print("Trained models saved to logreg_models.pkl")

if __name__ == "__main__":
    main()