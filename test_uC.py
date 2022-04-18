from NN.src.nn import  FFN
from NN.src.functions import *
from NN.src.optimizations import *
from NN.src.storage import *
from NN.src.train import *
from NN.src.numpy_package import numpy as np
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
weights_file = "modelGaussianMixture"

weights_file = "modelGaussianMixture" # name of the file
batch_size = 20
loss = train_batch(ann,Input,Target,
                    batch_size = batch_size,
                    weights_file=weights_file,
                    maxIterations = 1000,
                    minError =0.1,
                    minErrorDelta=0.01)

# individual optimization steps are also possible: ann.backwardProp(Input, Target)
# weights save automatically after training. Optional: save_weights(ann, weights_file)

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
Output = ann_evaluate.eval(Input[:batch_size]) # evaluate first batch

print("Predicted \t Real \t Error")
for i,pred in zip(Input[:,0],Output[:,0]):
    true_value = gaussian_mixture(i)
    pred_error = true_value - pred
    print("{0:.4f} \t {1:.4f} \t {2:.4f}".format(pred,true_value,pred_error))

# -------------------------------------------------
