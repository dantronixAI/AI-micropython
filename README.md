# AI-micropython
ANN implementation for micropython

Requirements:
 - ulab (https://github.com/v923z/micropython-ulab)

Optimization algorithms and parameters (mantaining the same parameters' name and default) are implemented as in: 
https://pytorch.org/docs/stable/optim.html#algorithms

- Available optimization algorithms:
  - rmsprop, sgd, adam, adamw, radam, adagrad

- Available activation Functions:
  - sigmoid, tanh, arctan, softplus, relu, lrelu, linear

The training can be perfomed in both: the microcontroller and the pc. However, due to memory capacity and slow processing time, training is recommened in the pc. The weights are saved after training, so they can be directly used in the microcontoller.

Full example for the PC and the microcontroller are also provided. The weights are saved in the folder NN/src/output

If the training is performed in the microcontroller:
  -Due to memory limitations, separate the data in mini batches, as presented in the example.

