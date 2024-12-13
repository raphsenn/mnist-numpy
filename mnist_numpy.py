import numpy as np


def relu(z: np.ndarray, derv: bool=False) -> np.ndarray:
    if derv: return np.where(z > 0, 1, 0) 
    return np.maximum(z, 0)


def softmax(z: np.ndarray) -> np.ndarray:
    return np.array([np.exp(z[i])/np.sum(np.exp(z)) for i in range(len(z))]) 


class BobNet:
    def __init__(self, n_in: int, n_hidden: int, n_out: int) -> None:
        self.w1 = np.random.rand(n_in, n_hidden)
        self.b1 = np.random.rand(n_hidden)
        self.w2 = np.random.rand(n_hidden, n_out)
        self.b2 = np.random.rand(n_out)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            lr: float=0.1,
            epochs: int=100, 
            batch_size: int=16,
            verbose: bool=True) -> None:
        N = X.shape[0] 

        # Simple implementation of stochastic gradient descent. 
        for epoch in range(epochs):
            # Shuffle dataset
            indices = np.random.permutation(N)
            X, y = X[indices], y[indices]    

            # Iterate over mini-batches 
            for i in range(0, N, batch_size):
                X_batch, y_batch = X[i::i+batch_size], y[i::i+batch_size]

                # Forward pass
                z1 = np.dot(X_batch, self.w1) + self.b1
                h1 = relu(z1)
                z2 = np.dot(h1, self.w2) + self.b2
                h2 = softmax(z2)

                # Backpropagation


    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.dot(x, self.w1) + self.b1
        x = relu(x)
        x = np.dot(x, self.w2) + self.b2
        return softmax(x)