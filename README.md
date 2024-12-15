# mnist-numpy
MNIST (Digits &amp; Fashion) Neural Network from scratch in raw NumPy, benchmarked against PyTorch.

## Using PyTorch

##### 2-Layer MLP
* 784 Input neurons
* 10 Hidden neurons
* 10 Output neurons

#### Using Stochastic Gradient Descent (lr=0.01) and Cross-Entropy:

##### Training:

```console
Epoch: 0	Loss: 24.33	Accuracy: 0.1121
Epoch: 10	Loss: 1.56	Accuracy: 0.4737
Epoch: 20	Loss: 1.15	Accuracy: 0.6191
Epoch: 30	Loss: 0.77	Accuracy: 0.7517
Epoch: 40	Loss: 0.49	Accuracy: 0.8508
Epoch: 50	Loss: 0.42	Accuracy: 0.8724
Epoch: 60	Loss: 0.33	Accuracy: 0.9003
Epoch: 70	Loss: 0.32	Accuracy: 0.9038
Epoch: 80	Loss: 0.28	Accuracy: 0.9169
Epoch: 90	Loss: 0.25	Accuracy: 0.9247
Epoch: 100	Loss: 0.29	Accuracy: 0.9098
Epoch: 110	Loss: 0.23	Accuracy: 0.9323
Epoch: 120	Loss: 0.22	Accuracy: 0.9347
Epoch: 130	Loss: 0.29	Accuracy: 0.9079
Epoch: 140	Loss: 0.20	Accuracy: 0.9400
Epoch: 150	Loss: 0.19	Accuracy: 0.9425
Epoch: 160	Loss: 0.18	Accuracy: 0.9444
Epoch: 170	Loss: 0.18	Accuracy: 0.9456
Epoch: 180	Loss: 0.18	Accuracy: 0.9455
Epoch: 190	Loss: 0.21	Accuracy: 0.9324
```

##### Testing:

```console
Accuracy on Testing set: 0.92
```


#### Using Adam (learning rate = 0.001) and Cross-Entropy:

##### 2-Layer MLP
* 784 Input neurons
* 512 Hidden neurons
* 10 Output neurons

##### Testing:

```console
Epoch: 0	Loss: 31.40	Accuracy: 0.0635
Epoch: 10	Loss: 4.52	Accuracy: 0.7986
Epoch: 20	Loss: 1.92	Accuracy: 0.8837
Epoch: 30	Loss: 0.99	Accuracy: 0.9108
Epoch: 40	Loss: 0.68	Accuracy: 0.9234
Epoch: 50	Loss: 0.49	Accuracy: 0.9323
Epoch: 60	Loss: 0.38	Accuracy: 0.9388
Epoch: 70	Loss: 0.30	Accuracy: 0.9455
Epoch: 80	Loss: 0.25	Accuracy: 0.9505
Epoch: 90	Loss: 0.21	Accuracy: 0.9556
Epoch: 100	Loss: 0.18	Accuracy: 0.9598
Epoch: 110	Loss: 0.16	Accuracy: 0.9635
Epoch: 120	Loss: 0.14	Accuracy: 0.9673
Epoch: 130	Loss: 0.12	Accuracy: 0.9707
Epoch: 140	Loss: 0.11	Accuracy: 0.9736
Epoch: 150	Loss: 0.10	Accuracy: 0.9764
Epoch: 160	Loss: 0.09	Accuracy: 0.9787
Epoch: 170	Loss: 0.08	Accuracy: 0.9806
Epoch: 180	Loss: 0.07	Accuracy: 0.9826
Epoch: 190	Loss: 0.06	Accuracy: 0.9845
```

##### Testing:

```console
Accuracy on Testing set: 0.97
```

## Using NumPy

##### 2-Layer MLP
* 784 Input neurons
* 10 Hidden neurons
* 10 Output neurons


##### Training:
```console
Epoch: 0	Cross-Entropy: 1.24	Accuracy: 0.59
Epoch: 10	Cross-Entropy: 0.52	Accuracy: 0.84
Epoch: 20	Cross-Entropy: 0.45	Accuracy: 0.87
Epoch: 30	Cross-Entropy: 0.42	Accuracy: 0.88
Epoch: 40	Cross-Entropy: 0.40	Accuracy: 0.88
Epoch: 50	Cross-Entropy: 0.39	Accuracy: 0.89
Epoch: 60	Cross-Entropy: 0.38	Accuracy: 0.89
Epoch: 70	Cross-Entropy: 0.37	Accuracy: 0.89
Epoch: 80	Cross-Entropy: 0.37	Accuracy: 0.90
Epoch: 90	Cross-Entropy: 0.36	Accuracy: 0.90
Epoch: 100	Cross-Entropy: 0.35	Accuracy: 0.90
Epoch: 110	Cross-Entropy: 0.35	Accuracy: 0.90
Epoch: 120	Cross-Entropy: 0.34	Accuracy: 0.90
Epoch: 130	Cross-Entropy: 0.34	Accuracy: 0.90
Epoch: 140	Cross-Entropy: 0.33	Accuracy: 0.90
Epoch: 150	Cross-Entropy: 0.33	Accuracy: 0.91
Epoch: 160	Cross-Entropy: 0.32	Accuracy: 0.91
Epoch: 170	Cross-Entropy: 0.32	Accuracy: 0.91
Epoch: 180	Cross-Entropy: 0.32	Accuracy: 0.91
Epoch: 190	Cross-Entropy: 0.32	Accuracy: 0.91
```

##### Testing:

```console
Accuracy on Testing set: 0.9118
```

## License
* MNIST Digits: Public domain, created by Yann LeCun and collaborators.
* Fashion-MNIST: MIT License, created by Zalando Research.