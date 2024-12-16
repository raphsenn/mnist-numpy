import torch
import torch.nn as nn
import torch.nn.functional as F


class BobNetTorchModel(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(n_in, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x


class BobNetTorch:
    def __init__(self, n_in: int, n_hidden: int, n_out: int) -> None:
        self.model = BobNetTorchModel(n_in=n_in, n_hidden=n_hidden, n_out=n_out)

    def fit(self, X: torch.Tensor, y: torch.Tensor, lr: float=0.1, epochs: int=100, batch_size: int=32, verbose: bool=True) -> None:
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr) 
        criterion = nn.CrossEntropyLoss() 

        # Stochastic Gradient Descent
        for epoch in range(epochs):
            indices = torch.randperm(X.shape[0])
            X, y = X[indices], y[indices] 
            
            for i in range(0, X.shape[0], batch_size):
                X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]

                # Clear gradients 
                optimizer.zero_grad()

                # Forward pass 
                y_hat = self.model.forward(X_batch)

                # Cross-Entropy loss
                loss = criterion(y_hat, y_batch.long()) 
                loss.backward() 
                optimizer.step()
            
            if verbose and epoch % 10 == 0:
                # Calculate accuracy
                y_hat = self.model.forward(X) 
                cross_entropy = criterion(y_hat, y.long()) 
                preds = torch.argmax(y_hat, dim=1)
                accuracy = (preds == y).float().mean() 
                print(f'Epoch: {epoch}\tCross-Entropy: {cross_entropy.item():.4f}\tAccuracy: {accuracy:.4f}')

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model.forward(x)
        out = torch.argmax(out, dim=1) 
        return out