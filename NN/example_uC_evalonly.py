from src.nn import  FFN
from src.functions import *
from src.storage import *
from src.numpy_package import numpy as np
import random
# ---------------------------------------------------------
# Available Optimization Classes
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

Input = np.zeros((N_points,1)) # N_points x 1
Input[:,0] = np.linspace(-10,10,N_points)


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

print("Predicted \t Real \t Error")

import time

for i in range(N_points):
    true_value = gaussian_mixture(Input[i])
    pred_value = ann_evaluate.eval(Input[i:i+1])[0]
    pred_error = true_value - pred_value # evaluate first batch
    print("{0:.4f} \t {1:.4f} \t {2:.4f}".format(pred_value[0],true_value[0],pred_error[0]))


# Measuring evaluation of N_points
ini_time = time.time_ns()
for i in range(N_points):
    ann_evaluate.eval(Input[i:i+1])[0]
end_time = time.time_ns()
total_time_seconds = (end_time-ini_time)/1e9

print("total time : {} s".format(total_time_seconds))
# -------------------------------------------------

