# AI-micropython
ANN implementation for micropython

Requirements:
 - ulab (https://github.com/v923z/micropython-ulab)

Optimization algorithms and parameters are implemented as in: 
https://pytorch.org/docs/stable/optim.html#algorithms

- Available optimization algorithms:
  - rmsprop, sgd, adam, adamw, radam, adagrad

- Available activation Functions:
  - sigmoid, tanh, arctan, softplus, relu, lrelu, linear

The training can be perfomed in both: the microcontroller and the pc. However, due to memory capacity and slow processing time, training is recommened in the pc. The weights are saved after training, so it can be copied to the microcontoller.



