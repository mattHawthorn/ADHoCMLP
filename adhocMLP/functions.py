#encoding: utf-8
from numpy import (array, ndarray, transpose, repeat, mean, std, arange, 
                   apply_along_axis, concatenate, unique, zeros, ones, dot, 
                   exp, log, tanh, arctan, square, sqrt, inf, maximum)
from numpy.random import randn, random, permutation
from scipy.stats import logistic as scipy_logistic
from scipy.stats import entropy
from functools import reduce
from itertools import product
from sklearn.preprocessing import OneHotEncoder


######################################################
## ADHoCNN:                                         ##
## Almost-Done Homebrew Categorical Neural Network  ##
## Or maybe Always-Developing...                    ##
######################################################



####################################################################
# common activation functions not defined directly in numpy or scipy

# scipy.stats.logistic is a class with CDF and other methods; we want the CDF
def logistic(x):
   return 1.0/(1.0+np.exp(-x))

# This a common output activation in regression problems
def identity(x):
    return x

# rectified linear unit
def ReLU(x):
    return maximum(0.0,x)

# smooth alternative to a ReLU
def softplus(x):
    return log(1.0+exp(x))

# And this one helps with categorical outputs
def softmax(x):
    e = exp(X)
    return e/e.sum(axis=1,keepdims=True)


###################################################################
# common loss functions.  These should take a true output and a 
# predicted output and return something of the same shape.
# I.e., mupltiple output columns are allowed, and errors are not 
# aggregated across rows.

def squared_loss(y,y_hat):
    e = 0.5*square(y_hat-y)
    if e.shape[1] == 1:
        return e
    else:
        return e.sum(axis=1,keepdims=True)
    
def cross_entropy(y,y_hat):
    return (-y*log(y_hat)).sum(axis=1,keepdims=True)


###################################################################
# Activation derivatives, in terms of the output.
# These should be written to apply to the cached values of the
# output of a neural network layer, returning the derivative of the
# output with respect to the weighted inputs.
# all of these will apply entry-wise to numpy arrays.

def logistic_deriv(y):
    return y*(1.0-y)
    
def arctan_deriv(y):
    return 1.0/(1.0+y*y)
    
def tanh_deriv(y):
    return 1.0 - y*y
    
def softplus_deriv(y):
    e = exp(x)
    return (e-1.0)/e
    
def identity_deriv(y):
    return 1.0
    
exp_deriv = identity

def ReLU_deriv(y):
    return (y>0.0).astype('float')


###################################################################
# Loss function derivatives.
# These should take predictions and ground truth as arguments;
# potentially saves a little computation over computing from errors

def squared_loss_deriv(y,y_hat):
    return y_hat-y
    
def cross_entropy_deriv(y,y_hat):
    return 

###################################################################
# "symbolic derivative" of a given function- pauper's version.
# Look up the name of the function in the global namespace and see
# if it has a name_deriv cousin.

def deriv(function):
    try:
        d = globals()[function.__name__ + '_deriv']
    except KeyError:
        raise KeyError("Illegal function: {}".format(function.__name__))
    
    return d

