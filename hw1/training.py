import numpy as np
import dataset
from _collections_abc import Sequence
from numpy.typing import NDArray

def train(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
          epochs: int, target: int, digits: Sequence[int] = range(0, 10), 
          verbose: bool = True) -> tuple[NDArray[np.float64], float]:
    """performs PLA training loop on MNIST data given digits and a target

    Args:
        X_train (np.ndarray): input vectors for training data
        y_train (np.ndarray): labels for each vector in training data
        X_val (np.ndarray): input vectors for validation data
        y_val (np.ndarray): labels for each vector in validation data
        epochs (int): number of times to train on the data
        target (int): digit the weights will train to recognize
        digits (Sequence[int], optional): digits that will be trained on; it is not enforced
        that digits has to contain target, which will result in trivial output. Defaults to range(0, 10).
        verbose (bool, optional): whether the function prints epoch updates on accuracy. Defaults to True.

    Returns:
        tuple[NDArray[np.float64], float]: tuple containing the best weights and accuracy according to validation data
    """
    # assert training target is defined
    if target is None:
        raise RuntimeError("Must select target digit to train for")
    
    # filter for the digits that we want the model to work with
    indices = np.flatnonzero(np.isin(y_train, digits))
    indices_val = np.flatnonzero(np.isin(y_val, digits))
    X_val_sub = X_val[indices_val]
    y_val_sub = np.where(y_val[indices_val] == target, 1, -1)
    
    y_train_yn = np.where(y_train == target, 1, -1)
    
    # create weights vector
    w = np.zeros(X_train.shape[1])
    
    # save some information for metrics and best performing weights
    best_w = np.array([])
    best_acc = 0.0
    val_accs = []
    
    # training loop, first epochs aka how many times the data is trained on
    for _i in range(epochs):
        # shuffle the data each time to attempt to enhance the optimum found
        # since the algorithm is sensitive to starting conditions
        indices = np.random.permutation(indices)
        
        # training over data
        # doing 1-by-1 but could do all at once, or even in batches
        for idx in indices:
            x = X_train[idx]
            y = y_train_yn[idx]
            
            # update rule
            if y * (x @ w) <= 0:
                w += y * x
        
        # calculating and saving validation accuracy
        acc = np.sum(y_val_sub * (X_val_sub @ w) > 0) / len(indices_val)
        val_accs.append(acc)
        if best_acc < acc:
            best_acc = acc
            best_w = w.copy()
        
        if verbose:
            print(f"Epoch: {_i + 1} \t Accuracy: {acc:.4f}")
    if verbose:
        print(f"Training Complete. Best score: {best_acc:.4f}")
    return best_w, best_acc

def train_ez(epochs: int, target: int, digits: Sequence[int] = range(0, 10),
          verbose: bool = True)  -> tuple[NDArray[np.float64], float]:   
    # get training data
    (X_train, y_train), (X_val, y_val) = dataset.load_mnist_ez()
    return train(X_train, y_train, X_val, y_val, epochs, target, digits, verbose)
     

if __name__ == '__main__':
    epochs = 15
    target = 3
    digits = range(5)
    w, acc = train_ez(epochs, target, digits)
    print(f"Best accuracy achieved: {acc}")
    
    



