import numpy as np


def ReLU(z: np.ndarray, derv: bool=False) -> np.ndarray:
    if derv: return np.where(z > 0, 1, 0) 
    return np.maximum(z, 0)


class BobNet:
    def __init__(self, n_in: int, n_hidden: int, n_out: int) -> None:
        self.w1 = np.random.rand(n_in, n_hidden)
        self.b1 = np.random.rand(n_hidden)
        self.w2 = np.random.rand(n_hidden, n_out)
        self.b2 = np.random.rand(n_out)

    def fit(X: np.ndarray, y: np.ndarray, lr: float=0.1, epochs: int=100, verbose: bool=True) -> None:
        pass

    def predict(x: np.ndarray) -> np.ndarray:
        pass