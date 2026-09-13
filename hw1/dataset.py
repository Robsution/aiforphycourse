import numpy as np
from pathlib import Path

def read_idx(path: str | Path) -> np.ndarray:
    """reads a single idx file with ubyte datatype

    Args:
        path (str | Path): path to idx file

    Returns:
        np.ndarray: returns array with samples based off of file-provided dimensions etc
    """
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



def load_mnist(data_dir: str | Path = "./data"):
    """loads the mnist dataset train and validation sets

    Args:
        data_dir (str | Path, optional): path to the mnist idx files. Defaults to "./data".

    Returns:
        _type_: tuple of tuple pairs with training and validation sets
    """
    path_base = Path(data_dir)
    files = {
        'train_images' : 'train-images-idx3-ubyte',
        'train_labels' : 'train-labels-idx1-ubyte',
        'val_images' : 't10k-images-idx3-ubyte',
        'val_labels' : 't10k-labels-idx1-ubyte',
    }
    X_train = read_idx(path_base / files['train_images'])
    y_train = read_idx(path_base / files['train_labels'])
    X_val = read_idx(path_base / files['val_images'])
    y_val = read_idx(path_base / files['val_labels'])
    
    return (X_train, y_train), (X_val, y_val)

        
        
