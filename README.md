# mnist-numpy
MNIST (Digits &amp; Fashion) Neural Network from scratch in raw NumPy, benchmarked against PyTorch.

## Using PyTorch

##### 2-Layer MLP
* 784 Input neurons
* 512 Hidden neurons
* 10 Output neurons

#### Using Stochastic Gradient Descent and Cross-Entropy:

##### Training (lr = 0.01):

```console
Epoch: 0	Cross-Entropy: 0.5878	Accuracy: 0.8890
Epoch: 10	Cross-Entropy: 0.0584	Accuracy: 0.9548
Epoch: 20	Cross-Entropy: 0.0695	Accuracy: 0.9729
Epoch: 30	Cross-Entropy: 0.0883	Accuracy: 0.9808
Epoch: 40	Cross-Entropy: 0.0425	Accuracy: 0.9867
```

##### Testing:

```console
Accuracy on Testing set: 0.9774
```

## Using NumPy

##### 2-Layer MLP
* 784 Input neurons
* 512 Hidden neurons
* 10 Output neurons


##### Training (lr = 0.0004):
```console
Epoch: 0	Cross-Entropy: 0.3087	Accuracy: 0.9129
Epoch: 10	Cross-Entropy: 0.1689	Accuracy: 0.9518
Epoch: 20	Cross-Entropy: 0.1344	Accuracy: 0.9617
Epoch: 30	Cross-Entropy: 0.1180	Accuracy: 0.9657
Epoch: 40	Cross-Entropy: 0.1046	Accuracy: 0.9698
```

##### Testing:

```console
Accuracy on Testing set: 0.9573
```

## License
* MNIST Digits: Public domain, created by Yann LeCun and collaborators.
* Fashion-MNIST: MIT License, created by Zalando Research.