import numpy as np
import dataset

def train(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, 
          y_val: np.ndarray, epochs: int, target: int, digits: list = range(0, 10),
          verbose: bool = True):
    # assert training mode is selected and not xcl training mode
    if target is None:
        raise RuntimeError("Must select target digit to train for")

    indices = np.flatnonzero(np.isin(y_train, digits))
    indices_val = np.flatnonzero(np.isin(y_val, digits))
    X_val_sub = X_val[indices_val]
    y_val_sub = np.where(y_val[indices_val] == target, 1, -1)
    
    y_train_yn = np.where(y_train == target, 1, -1)
    
    # create weights vector
    w = np.zeros(X_train.shape[1])
    # learning rate
    lr = 1.0
    
    best_w = None
    best_acc = 0.0
    val_accs = []
    for _i in range(epochs):
        indices = np.random.permutation(indices)
        for idx in indices:
            x = X_train[idx]
            y = y_train_yn[idx]
            
            if y * (x @ w) <= 0:
                w += lr * y * x
        
        # validation accuracy
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

    
if __name__ == '__main__':
    # get training data
    (X_train, y_train), (X_val, y_val) = dataset.load_mnist_ez()
    
    epochs = 15
    digits = range(5)
    target = 3
    
    train(X_train, y_train, X_val, y_val, epochs, target, digits)



