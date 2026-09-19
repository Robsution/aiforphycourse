from training import train
import dataset
from itertools import combinations
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent / "figures"


def pairwise(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, 
          y_val: np.ndarray, epochs: int) -> None:
    """displays the matrix of accuracy of training between two pairs of digits, 0-9.

    Args:
        X_train (np.ndarray): input vectors of training data
        y_train (np.ndarray): labels of training data
        X_val (np.ndarray): input vectors of validation data
        y_val (np.ndarray): labels of training data
        epochs (int): number of times to train for each pair
    """
    # initialize score matrix
    scores = np.zeros((10, 10))

    # train on all possible combinations of digits and save it to score
    # combinations order does not matter
    for comb in combinations(range(10), 2):
        digits = list(comb)
        target = digits[0]
        _, pair_acc = train(X_train, y_train, X_val, y_val, epochs, target, digits, verbose = False)
        scores[comb] = pair_acc

    # since combinations order does not matter, we make the matrix symmetric to display 
    # the full data, adding a diagonal of ones for trivial training
    scores = np.triu(scores) + np.triu(scores).T + np.diag(np.ones(10))

    # plotting and making it look nice
    im = plt.imshow(scores, cmap = 'viridis')
    plt.title("Pairwise Prediction Accuracy")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar(im)

    plt.xlabel("Negative digit")
    plt.ylabel("Positive digit")
    
    # saves figure
    plt.savefig(PLOT_DIR / "pairwise_preds.png")
    
    print("Saved pairwise plot")


def allvsone(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, 
          y_val: np.ndarray, epochs: int) -> None:
    """makes a table of all vs. one training for all digits

    Args:
        X_train (np.ndarray): input vectors of training data
        y_train (np.ndarray): labels of training data
        X_val (np.ndarray): input vectors of validation data
        y_val (np.ndarray): labels of training data
        epochs (int): number of times to train for each pair
    """
    # create table to store scores
    scores = np.zeros(10)
    # list of digits
    digits = range(10)
    
    # train each digit against all, save scores
    for target in digits:
        _, all_acc = train(X_train, y_train, X_val, y_val, epochs, target, digits, verbose = False)
        scores[target] = all_acc
        
    # plotting and making the table
    fig, ax = plt.subplots(figsize=(10, 1.8))
    ax.axis('off')
    
    col_labels = [f"Digit {d}" for d in digits]
    cell_text = [[f"{s*100:.2f}%" for s in scores]]
    
    table = ax.table(
        cellText= cell_text,
        colLabels= col_labels,
        loc='center',
        cellLoc='center',
        bbox=Bbox.from_bounds(0.0, 0.0, 1.0, 0.8)
    )
    
    table.scale(1.0, 2.0)
    table.set_fontsize(12)
    
    plt.tight_layout
    ax.set_title("All vs. One digit-wise prediction accuracies", fontsize = 13, pad = 2)
    
    # save the figure
    plt.savefig(PLOT_DIR / "all_preds.png",  dpi=300, bbox_inches='tight')
    
    print("Saved all vs. one table")


def confidence_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
) -> None:
  """evaluates 10-class prediction using geometric confidence margins and selective classification.

  Args:
      X_train (np.ndarray): input vectors of training data
      y_train (np.ndarray): labels of training data
      X_val (np.ndarray): input vectors of validation data
      y_val (np.ndarray): labels of validation data
      epochs (int): number of times to train for each digit
  """
  digits = range(10)

  # train one classifier per digit against all others and collect weights
  weights = []
  for target in digits:
    w, _ = train(
        X_train, y_train, X_val, y_val, epochs, target, digits, verbose=False
    )
    weights.append(w)
  weights = np.array(weights)

  # ensure X_val is flattened and matches weight dimensions (with bias if needed)
  X_val_flat = X_val.reshape(X_val.shape[0], -1)
  if weights.shape[1] == X_val_flat.shape[1] + 1:
    X_eval = np.c_[np.ones(X_val_flat.shape[0]), X_val_flat]
    feature_norms = np.linalg.norm(weights[:, 1:], axis=1, keepdims=True)
  else:
    X_eval = X_val_flat
    feature_norms = np.linalg.norm(weights, axis=1, keepdims=True)

  # normalize weights by geometric feature norm: w / ||w||
  feature_norms[feature_norms == 0] = 1.0
  normalized_weights = weights / feature_norms

  # compute geometric confidence scores: (N, D) @ (D, 10) -> (N, 10)
  confidence_preds = X_eval @ normalized_weights.T
  preds = np.argmax(confidence_preds, axis=1)
  is_correct = preds == y_val
  overall_acc = np.mean(is_correct) * 100

  # calculate margin between winning class and runner-up
  sorted_scores = np.sort(confidence_preds, axis=1)
  margins = sorted_scores[:, -1] - sorted_scores[:, -2]

  # evaluate selective classification (accuracy vs coverage)
  thresholds = np.linspace(0, np.percentile(margins, 95), 50)
  coverages, accuracies = [], []
  for t in thresholds:
    mask = margins >= t
    if np.sum(mask) > 0:
      coverages.append(np.mean(mask) * 100)
      accuracies.append(np.mean(is_correct[mask]) * 100)

  # plotting margin distribution and coverage curve
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

  # plot A: margin distribution
  axes[0].hist(
      margins[is_correct], bins=30, alpha=0.6, label="Correct", density=True
  )
  axes[0].hist(
      margins[~is_correct],
      bins=30,
      alpha=0.6,
      label="Incorrect",
      density=True,
  )
  axes[0].set_title("Margin Distribution (Top − Runner-Up)")
  axes[0].set_xlabel("Score Margin ($w^T x / ||w||$)")
  axes[0].set_ylabel("Density")
  axes[0].legend()
  axes[0].grid(alpha=0.3)

  # plot B: accuracy vs data coverage
  axes[1].plot(coverages, accuracies, lw=2, label="Filtered Accuracy")
  axes[1].axhline(
      overall_acc,
      color="gray",
      linestyle="--",
      label=f"Baseline Acc ({overall_acc:.1f}%)",
  )
  axes[1].set_title("Accuracy vs. Data Coverage")
  axes[1].set_xlabel("% Samples Retained (Higher Threshold → Lower Coverage)")
  axes[1].set_ylabel("Accuracy on Retained Samples (%)")
  axes[1].invert_xaxis()
  axes[1].legend()
  axes[1].grid(alpha=0.3)

  fig.suptitle(
      f"10-Digit Confidence Predictor (Overall Accuracy: {overall_acc:.2f}%)",
      fontsize=13,
      fontweight="bold",
  )
  plt.tight_layout()

  # save the figure
  plt.savefig(PLOT_DIR / "confidence_eval.png", dpi=300, bbox_inches="tight")
  
  print("Saved confidence evaluation plot")

if __name__ == '__main__':
    # get easy training data
    (X_train, y_train), (X_val, y_val) = dataset.load_mnist_ez()
    epochs = 100
    
    pairwise(X_train, y_train, X_val, y_val, epochs)
    allvsone(X_train, y_train, X_val, y_val, epochs)
    confidence_eval(X_train, y_train, X_val, y_val, epochs)
    print("Done making figures.")