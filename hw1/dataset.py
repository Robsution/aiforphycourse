import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

def read_idx(path: str | Path) -> np.ndarray:
    """reads a single idx file with ubyte datatype

    Args:
        path (str | Path): path to idx file

    Returns:
        np.ndarray: returns array with samples based off of file-provided dimensions etc
    """
    # open data file
    with open(path,'rb') as f:
            # Read 'magic' bytes
            magic = f.read(4)
            # type of data
            type_code = magic[2]
            # number of dimensions
            nd = magic[3]
            dims = [ int.from_bytes(f.read(4), byteorder='big', signed=False) for _ in range(nd) ]
            
            # read rest of file
            file_data = np.frombuffer(f.read(), dtype=np.ubyte)
            # reshape into each individual image
            file_data = file_data.reshape(dims)
            return file_data



def load_mnist(data_dir: str | Path = DATA_DIR) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """loads the mnist dataset train and validation sets

    Args:
        data_dir (str | Path, optional): path to the mnist idx files. Defaults to "./data".

    Returns:
        _type_: tuple of tuple pairs with training and validation sets
    """
    # find path
    path_base = Path(data_dir)
    # default naming of mnist data
    files = {
        'train_images' : 'train-images-idx3-ubyte',
        'train_labels' : 'train-labels-idx1-ubyte',
        'val_images' : 't10k-images-idx3-ubyte',
        'val_labels' : 't10k-labels-idx1-ubyte',
    }
    # load files into memory
    X_train = read_idx(path_base / files['train_images'])
    y_train = read_idx(path_base / files['train_labels'])
    X_val = read_idx(path_base / files['val_images'])
    y_val = read_idx(path_base / files['val_labels'])
    
    return (X_train, y_train), (X_val, y_val)


def load_mnist_ez(data_dir: str | Path = DATA_DIR) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """easy version of load_mnist
    fetches mnist data, flattens it and adds a bias column

    Args:
        data_dir (str | Path, optional): path to the mnist data files. Defaults to "./data".

    Returns:
        _type_: tuple of tuple pairs with training and validation sets
    """
    # load dataset into memory
    (X_train, y_train), (X_val, y_val) = load_mnist(data_dir)
    
    # flatten and normalize
    X_train = X_train.reshape(X_train.shape[0], -1) / 255
    X_val = X_val.reshape(X_val.shape[0], -1) / 255
    # add bias x_0
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_val = np.c_[np.ones(X_val.shape[0]), X_val]
    
    return (X_train, y_train), (X_val, y_val)
    
        
