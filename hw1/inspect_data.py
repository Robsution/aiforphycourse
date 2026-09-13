import matplotlib.pyplot as plt
import numpy as np
import dataset


def plot_digit(image: np.ndarray, label: int):
    plt.imshow(image, cmap="gray")
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val) = dataset.load_mnist()
    i = 407
    plot_digit(X_train[i], y_train[i])