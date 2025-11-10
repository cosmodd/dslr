"""
Logistic Regression Basics: https://www.youtube.com/playlist?list=PLuhqtP7jdD8Chy7QIo5U0zzKP8-emLdny

- Binary Classification: Predicts 1 or 0. For multi-class, use One-vs-Rest: train one model per class.
- Linear Part (z): z = w1x1 + w2x2 + ... + wnxn + b = W * X + b (weights W times features X, plus bias b).
- Prediction (a): Sigmoid squashes z to [0,1] (better for probabilities): a = 1 / (1 + e^(-z)).
"""

"""
Cost Function: Log Loss https://community.deeplearning.ai/t/why-mse-is-not-a-good-loss-function-for-logistic-regression/255547

L = -(1/N) Σ [y log(a) + (1-y) log(1-a)] | (N = samples, y = true label 0/1, a = predicted prob)

- If y=1: Loss = -log(a). a≈1 -> low loss (good); a≈0 -> huge loss (bad!).
- If y=0: Loss = -log(1-a). a≈0 -> low; a≈1 -> huge.

Goal: Minimize L (perfect fit = L=0).
"""

"""
Training: Gradient Descent

- Gradients: Partial derivatives (dL/dW, dL/db) show how changing weights/bias affects loss, they point uphill.
- Update Rule: Step opposite the gradient, scaled by learning rate alpha (small step size to avoid overshooting).  
  W -= alpha * (dL/dW)  
  b -= alpha * (dL/db)

Deriving Gradients (using chain rule: dL/dW = dL/da x da/dz x dz/dW):

1. dL/da: How loss changes with prediction?  
   dL/da = -y/a + (1-y)/(1-a) = (a - y)/[a(1-a)]

2. da/dz: Sigmoid's slope (How prediction changes with z)
   da/dz = a(1-a)  (easy form—no need for raw e^(-z)).
   Deep dive: a = 1/(1+e^(-z)) -> da/dz = a^2 e^(-z)/(1+e^(-z))^2 = a(1-a).

3. dz/dW: How z changes with weights? dz/dW = X (features).
dz/db = 1 (bias affects z directly).

4. Combine for dL/dW:  
   dL/dW = (dL/da) x (da/dz) x (dz/dW) = [(a - y)/[a(1-a)]] x [a(1-a)] x X = (a - y) X  

5. For dL/db: Same but x1: dL/db = (a - y).  

Iterate: Predict → Compute loss/grads → Update → Repeat.
"""
import sys
import pandas as pd
import numpy as np
import pickle

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def load_data(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df.set_index('Index')
    print(f"Data loaded from {dataset_path}. Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
    print(f"----------------- Data Preview ------------------\n{df.head()}\n-------------------------------------------------")
    return df

def preprocess_data(df):
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
    # For now drop rows with NaN values, can impute them instead
    df = df.dropna()
    print(f"----------- Preprocessed Data Preview -----------\n{df.head()}\n-------------------------------------------------")
    return df

def scale_features(X):
    """
    feature values not scaled similarly = hurts convergence. (Ex: Arithmancy..) -> standardization:
    - substract mean (center around 0, negative values allowed)
    - divide by std deviation (scale to unit variance) -> better spread
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / (X_std)
    return X_scaled, X_mean, X_std

def binary_model(X, Y, learning_rate, epochs):
    m = X.shape[0]  # number of samples
    n = X.shape[1]  # number of features
    W = np.zeros(n)
    b = 0

    for epoch in range(epochs):
        Z = np.dot(X, W) + b # w1x1 + w2x2 + ... + wnxn + b
        A = sigmoid(Z)

        # Compute gradients & update parameters
        dW = (1/m) * np.dot(X.T, (A - Y))
        db = (1/m) * np.sum(A - Y)

        W -= learning_rate * dW
        b -= learning_rate * db

        if epoch % 100 == 0:
            # Compute loss for monitoring (add small value to avoid log(0) issue)
            loss = - (1/m) * np.sum(Y * np.log(A + 1e-15) + (1 - Y) * np.log(1 - A + 1e-15)) 
            print(f"Binary Model - Epoch {epoch}, Loss: {loss:.4f}")
    return W, b

def train_ovr(X, Y, learning_rate, epochs):
    classes = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    num_classes = len(classes)
    models = []
    for k in range(num_classes):
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
    X = preprocessed_df.drop(columns=['Hogwarts House']).values
    Y = preprocessed_df['Hogwarts House'].values
    print(f"Train X shape: {X.shape}, Train Y shape: {Y.shape}")

    X_scaled, X_mean, X_std = scale_features(X)

    models = train_ovr(X_scaled, Y, learning_rate=0.01, epochs=100000)

    save_data = {
        'models': models,
        'X_mean': X_mean,
        'X_std': X_std
    }

    with open('logreg_models.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    print("Trained models saved to logreg_models.pkl")

if __name__ == "__main__":
    main()