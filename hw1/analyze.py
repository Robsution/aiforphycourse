from training import train
import dataset
from itertools import combinations
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent / "figures"


def pairwise(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, 
          y_val: np.ndarray, epochs: int):
    scores = np.zeros((10, 10))

    for comb in combinations(range(10), 2):
        digits = list(comb)
        target = digits[0]
        _, pair_acc = train(X_train, y_train, X_val, y_val, epochs, target, digits, verbose = False)
        scores[comb] = pair_acc

    scores = np.triu(scores) + np.triu(scores).T + np.diag(np.ones(10))

    im = plt.imshow(scores, cmap = 'viridis')
    plt.title("Pairwise Prediction Accuracy")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar(im)

    plt.xlabel("Negative digit")
    plt.ylabel("Positive digit")

    plt.savefig(PLOT_DIR / "pairwise_preds.png")

def allvsone(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, 
          y_val: np.ndarray, epochs: int):
    scores = np.zeros(10)
    digits = range(10)
    
    for target in digits:
        _, all_acc = train(X_train, y_train, X_val, y_val, epochs, target, digits, verbose = False)
        scores[target] = all_acc
        
    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.axis("off")
    
    col_labels = [f"Digit {d}" for d in digits]
    cell_text = [[f"{s*100:.2f}%" for s in scores]]
    
    table = ax.table(
        cellText= cell_text,
        colLabels= col_labels,
        loc="center",
        cellLoc="center",
    )
    
    table.scale(1.0, 2.0)
    table.set_fontsize(12)
    
    plt.tight_layout
    plt.savefig(PLOT_DIR / "all_preds.png")


if __name__ == '__main__':
    # get easy training data
    (X_train, y_train), (X_val, y_val) = dataset.load_mnist_ez()
    epochs = 15
    
    pairwise(X_train, y_train, X_val, y_val, epochs)
    allvsone(X_train, y_train, X_val, y_val, epochs)
    print("Done making figures.")