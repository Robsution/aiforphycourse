import numpy as np
import dataset

def train(epochs: int, target: int, digits: list = range(0, 10)):
    # assert training mode is selected and not xcl training mode
    if target is None:
        raise RuntimeError("Must select target digit to train for")
        
    # get training data
    (X_train, y_train), (X_val, y_val) = dataset.load_mnist()
    # flatten and normalize
    X_train = X_train.reshape(X_train.shape[0], -1) / 255
    X_val = X_val.reshape(X_val.shape[0], -1) / 255
    # add bias x_0
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_val = np.c_[np.ones(X_val.shape[0]), X_val]

    indices = np.flatnonzero(np.isin(y_train, digits))
    indices_val = np.flatnonzero(np.isin(y_val, digits))
    X_val_sub = X_val[indices_val]
    y_val_sub = np.where(y_val[indices_val] == target, 1, -1)
    
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
            y = 1 if y_train[idx] == target else -1
            
            if y * (x @ w) <= 0:
                w += lr * y * x
        
        # validation accuracy
        acc = np.sum(y_val_sub * (X_val_sub @ w) > 0) / len(indices_val)
        val_accs.append(acc)
        print(f"Epoch: {_i + 1} \t Accuracy: {acc:4f}")
        
                
            
    
if __name__ == '__main__':
    epochs = 15
    digits = [8, 3]
    target = digits[0]
    train(epochs, target, digits)



