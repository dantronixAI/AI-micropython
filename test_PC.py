from NN.src.nn import  FFN
from NN.src.functions import *
from NN.src.optimizations import *
from NN.src.storage import *
from NN.src.train import *
from NN.src.numpy_package import numpy as np
import random
# ---------------------------------------------------------
# Available Optimization Functions
# ---------------------------------------------------------
# rmsprop, sgd, adam, adamw, radam, adagrad
# ---------------------------------------------------------
# ---------------------------------------------------------
# Available Activation Functions
# ---------------------------------------------------------
# sigmoid, tanh, arctan, softplus, relu, lrelu, linear
# ---------------------------------------------------------


# ---------------------------------------------------------
# Generating Data For training
# ---------------------------------------------------------
def gaussian_model(x, A=1, sigma=1, centre=0):
    return A*np.exp(-(x - centre)**2 / (2*sigma**2))
def gaussian_mixture(x ):
    return gaussian_model(x, 0.8, 0.5, -1.5) + gaussian_model(x, 1, 2, 0) + gaussian_model(x,  3, 0.4, 1.8)

N_points = 500

x = [(random.random()-0.5)*10 for i in range(N_points)]
Input = np.zeros((N_points,1)) # N_points x 1
Input[:,0] = x

Target = gaussian_mixture(Input) # N_points x 1
# ---------------------------------------------------------


# ---------------------------------------------------------
# Training Task
# ---------------------------------------------------------
params ={}
params['shape']=(1, 20,20, 1)  # (# inputs, # nn hidden_layer, ..., # nn hidden_layer, # outputs)
# activation and optimization are optional. Default:
# activation: sigmoid -- optimization: rmsprop
params['activation']={
    'layer1': tanh, # activation hidden layer1
    'layer2': tanh, # activation hidden layer2
    'layer3': relu, # output
}
params['optimization']=adam(lr=0.01)



ann = FFN(params)
weights_file = "modelGaussianMixture" # name of the file
loss = train(ann,Input,Target,
      weights_file=weights_file,
      maxIterations = 100000,
      minError =1e-4,
      minErrorDelta=1e-6)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
params ={}
params['shape']=(1, 20,20, 1) # (# inputs, # nn hidden_layer, ..., # nn hidden_layer, # outputs)
params['activation']={
    'layer1': tanh, # activation hidden layer1
    'layer2': tanh, # activation hidden layer2
    'layer3': relu, # output
}
weights_file = "modelGaussianMixture" # name of the file
ann_evaluate = FFN(params) # initialize the nn
restore_weights(ann_evaluate,weights_file) # restore the weights

Output = ann_evaluate.eval(Input) # eval requires 2d-array
# -------------------------------------------------


# -------------------------------------------------
# Visual Verification (pc only)
# -------------------------------------------------
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2,1,1)
x = np.linspace(-10,10,N_points)
plt.plot(x,gaussian_mixture(x),label="Target")
plt.plot(x,ann_evaluate.eval(x[None,:].T),'--',label="ANN")
plt.legend()

plt.subplot(2,1,2)
plt.plot(loss,label="loss")
plt.xlabel("epoch")
plt.legend()
# -------------------------------------------------
