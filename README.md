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
Epoch: 0	Cross-Entropy: 0.4144	Accuracy: 0.8893
Epoch: 10	Cross-Entropy: 0.1618	Accuracy: 0.9552
Epoch: 20	Cross-Entropy: 0.1011	Accuracy: 0.9722
Epoch: 30	Cross-Entropy: 0.0714	Accuracy: 0.9811
Epoch: 40	Cross-Entropy: 0.0542	Accuracy: 0.9861
Epoch: 50	Cross-Entropy: 0.0422	Accuracy: 0.9901
Epoch: 60	Cross-Entropy: 0.0346	Accuracy: 0.9925
Epoch: 70	Cross-Entropy: 0.0278	Accuracy: 0.9946
Epoch: 80	Cross-Entropy: 0.0234	Accuracy: 0.9956
Epoch: 90	Cross-Entropy: 0.0194	Accuracy: 0.9970
```

##### Testing:

```console
Accuracy on Testing set: 0.9812
```

## Using NumPy

##### 2-Layer MLP
* 784 Input neurons
* 512 Hidden neurons
* 10 Output neurons


##### Training (lr = 0.095):
```console
Epoch: 0	Cross-Entropy: 0.3059	Accuracy: 0.9133
Epoch: 10	Cross-Entropy: 0.1696	Accuracy: 0.9520
Epoch: 20	Cross-Entropy: 0.1355	Accuracy: 0.9615
Epoch: 30	Cross-Entropy: 0.1185	Accuracy: 0.9666
Epoch: 40	Cross-Entropy: 0.1089	Accuracy: 0.9685
Epoch: 50	Cross-Entropy: 0.0996	Accuracy: 0.9716
Epoch: 60	Cross-Entropy: 0.0920	Accuracy: 0.9743
Epoch: 70	Cross-Entropy: 0.0884	Accuracy: 0.9746
Epoch: 80	Cross-Entropy: 0.0824	Accuracy: 0.9767
Epoch: 90	Cross-Entropy: 0.0881	Accuracy: 0.9727
```

##### Testing:

```console
Accuracy on Testing set: 0.9615
```

## License
* MNIST Digits: Public domain, created by Yann LeCun and collaborators.
* Fashion-MNIST: MIT License, created by Zalando Research.