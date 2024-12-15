# mnist-numpy
MNIST (Digits &amp; Fashion) Neural Network from scratch in raw NumPy, benchmarked against PyTorch.

## Using PyTorch

##### 2-Layer MLP
* 784 Input neurons
* 10 Hidden neurons
* 10 Output neurons

#### Using Stochastic Gradient Descent and Cross-Entropy:

##### Training (lr = 0.01):

```console
Epoch: 0	Cross-Entropy: 0.4523	Accuracy: 0.8723
Epoch: 10	Cross-Entropy: 0.1518	Accuracy: 0.9266
Epoch: 20	Cross-Entropy: 0.3325	Accuracy: 0.9350
Epoch: 30	Cross-Entropy: 0.1523	Accuracy: 0.9398
Epoch: 40	Cross-Entropy: 0.0954	Accuracy: 0.9429
Epoch: 50	Cross-Entropy: 0.2030	Accuracy: 0.9458
Epoch: 60	Cross-Entropy: 0.2093	Accuracy: 0.9459
Epoch: 70	Cross-Entropy: 0.1581	Accuracy: 0.9483
Epoch: 80	Cross-Entropy: 0.1884	Accuracy: 0.9496
Epoch: 90	Cross-Entropy: 0.2698	Accuracy: 0.9506
```

##### Testing:

```console
Accuracy on Testing set: 0.9403
```

## Using NumPy

##### 2-Layer MLP
* 784 Input neurons
* 10 Hidden neurons
* 10 Output neurons


##### Training (lr = 0.0004):
```console
Epoch: 0	Cross-Entropy: 0.5373	Accuracy: 0.8453
Epoch: 10	Cross-Entropy: 0.3577	Accuracy: 0.8984
Epoch: 20	Cross-Entropy: 0.3469	Accuracy: 0.9010
Epoch: 30	Cross-Entropy: 0.3321	Accuracy: 0.9070
Epoch: 40	Cross-Entropy: 0.3165	Accuracy: 0.9094
Epoch: 50	Cross-Entropy: 0.3082	Accuracy: 0.9114
Epoch: 60	Cross-Entropy: 0.2998	Accuracy: 0.9144
Epoch: 70	Cross-Entropy: 0.2998	Accuracy: 0.9131
Epoch: 80	Cross-Entropy: 0.2911	Accuracy: 0.9169
Epoch: 90	Cross-Entropy: 0.2882	Accuracy: 0.9158
```

##### Testing:

```console
Accuracy on Testing set: 0.9153
```

## License
* MNIST Digits: Public domain, created by Yann LeCun and collaborators.
* Fashion-MNIST: MIT License, created by Zalando Research.