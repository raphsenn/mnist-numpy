import numpy as np


def relu(z: np.ndarray, derv: bool=False) -> np.ndarray:
    if derv: return np.where(z > 0, 1, 0) 
    return np.maximum(z, 0)


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int=10) -> np.ndarray:
    y_hot = np.zeros((len(y), num_classes))
    y_hot[range(y.shape[0]), y] = 1
    return y_hot


class BobNet:
    def __init__(self, n_in: int, n_hidden: int, n_out: int) -> None:
        # He Initialization
        self.w1 = np.random.normal(0, np.sqrt(2/n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.w2 = np.random.normal(0, np.sqrt(2/n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            lr: float=0.1,
            epochs: int=100, 
            batch_size: int=32,
            verbose: bool=True) -> None:
        N = X.shape[0] 

        # Simple implementation of Stochastic Gradient Descent
        for epoch in range(epochs):
            # Shuffle dataset
            indices = np.random.permutation(N)
            X, y = X[indices], y[indices]    
            y_hot = one_hot(y)

            for i in range(0, N, batch_size):
                X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]
                y_batch_hot = one_hot(y_batch, num_classes=self.w2.shape[1])

                # Forward pass
                z1 = np.dot(X_batch, self.w1) + self.b1                     # N x 512
                h1 = relu(z1)                                               # N x 512
                z2 = np.dot(h1, self.w2) + self.b2                          # N x 10
                h2 = softmax(z2)                                            # N x 10

                # Backpropagation
                dz2 = h2 - y_batch_hot                                      # N x 10
                dw2 = np.dot(h1.T, dz2)                                     # 512 x 10
                db2 = np.sum(dz2, axis=0)

                dh1 = np.dot(dz2, self.w2.T)                                # N x 512
                dz1 = dh1 * relu(z1, derv=True)                             # N x 512
                dw1 = np.dot(X_batch.T, dh1)                                # 784 x 512
                db1 = np.sum(dz1, axis=0)

                # Update Parameters
                self.w2 = self.w2 - lr * dw2
                self.b2 = self.b2 - lr * db2
                self.w1 = self.w1 - lr * dw1
                self.b1 = self.b1 - lr * db1

            if verbose and epoch % 10 == 0:
                y_hat = self.forward(X)
                cross_entropy = - np.sum(y_hot * np.log(y_hat))/N
                y_hat = np.argmax(y_hat, 1) 
                accuracy = (y_hat == y).mean()
                print(f'Epoch: {epoch}\tCross-Entropy: {cross_entropy:.4f}\tAccuracy: {accuracy:.4f}')

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.dot(x, self.w1) + self.b1
        x = relu(x)
        x = np.dot(x, self.w2) + self.b2
        x = softmax(x)
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = self.forward(x)
        out = np.argmax(out, 1)
        return out