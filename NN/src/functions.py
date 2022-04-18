from .numpy_package import numpy as np
def sigmoid(x, Derivative=False):
    if not Derivative:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        f = sigmoid(x)
        return f* (1.0 -f)

def tanh(x, Derivative=False):
    if not Derivative:
        return np.tanh(x)
    else:
        f = np.tanh(x)
        return 1.0-f*f

def arctan(x, Derivative=False):
    if not Derivative:
        return np.arctan(x)
    else:
        return 1.0/(1.0+x*x)

def softplus(x, Derivative=False):
    if not Derivative:
        return np.log(1+np.exp(x))
    else:
        return 1/(1+np.exp(x))

def linear(x, Derivative=False):
    if not Derivative:
        return x
    else:
        return np.ones(x.shape)

def relu(x, Derivative=False):
    rows,cols = x.shape
    for row in range(rows):
        if not Derivative:
            mask_neg = x[row, :] < 0.0
            x[row][mask_neg] = 0.0
        else:
            mask_pos = x[row, :] >= 0.0
            mask_neg = x[row, :] < 0.0
            x[row][mask_pos] = 1.0
            x[row][mask_neg] = 0.0
    return x

def lrelu (x, Derivative=False):
    rows,cols = x.shape
    for row in range(rows):
        if not Derivative:
            mask_neg = x[row, :] < 0.0
            x[row][mask_neg] = 0.1*x[row][mask_neg]
        else:
            mask_pos = x[row, :] >= 0.0
            mask_neg = x[row, :] < 0.0
            x[row][mask_pos] = 1.0
            x[row][mask_neg] = 0.1
    return x


