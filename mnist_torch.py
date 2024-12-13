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
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out

    def fit(self, X: torch.Tensor, y: torch.Tensor, lr: float=0.1, epochs: int=100, verbose: bool=True) -> None:
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr) 
        criterion = nn.CrossEntropyLoss() 
        y_one_hot = F.one_hot(y.long(), num_classes=self.n_out).float()
        for epoch in range(epochs):
            self.model.train() 
            y_hat = self.model.forward(X)
            optimizer.zero_grad() 
            loss = criterion(y_hat, y_one_hot) 
            loss.backward() 
            optimizer.step()
            if verbose and epoch % 10 == 0:
                # Calculate accuracy
                preds = torch.argmax(y_hat, dim=1)
                accuracy = (preds == y).float().mean() 
                print(f'Epoch: {epoch}\tLoss: {loss.item():.2f}\tAccuracy: {accuracy:.4f}')

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)