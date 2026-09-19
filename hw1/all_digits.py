from pathlib import Path
import numpy as np
from dataset import load_mnist
from training import train_ez

digits = range(10)
epochs = 1000

# collect weights into a 2D matrix of shape (10, D)
weights = np.array([train_ez(epochs, i, digits, verbose=False)[0] for i in digits])

# load and preprocess validation data
DATA_DIR = Path(__file__).resolve().parent / "data"
(_, _), (X_val, y_val) = load_mnist()
X_val = X_val.reshape(X_val.shape[0], -1) / 255.0
X_val = np.c_[np.ones(X_val.shape[0]), X_val]  # Shape: (N, D)

# compute geometric confidence: (w^T x) / ||w_features||
# exclude the prepended bias column (index 0) from the norm calculation
feature_norms = np.linalg.norm(weights[:, 1:], axis=1, keepdims=True)
feature_norms[feature_norms == 0] = 1.0  # Prevent division by zero
normalized_weights = weights / feature_norms

# matrix multiplication: (N, D) @ (D, 10) -> (N, 10)
confidence_preds = X_val @ normalized_weights.T

# Pick the class with the highest score per sample
preds = np.argmax(confidence_preds, axis=1)

# Compute accuracy
accuracy = np.mean(preds == y_val)
print(f"Accuracy: {accuracy * 100:.2f}%")