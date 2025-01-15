import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
train_samples = train_x_orig.shape[0]
num_pixels = train_x_orig.shape[1]
test_samples = test_x_orig.shape[0]

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

def model(I, O, layers_dims, learning_rate = 0.0075, num_iterations = 3000):
  grads = {}
  costs = []
  m = I.shape[1]
  (n_x, n_h, n_y) = layers_dims

  pars = init_pars(n_x, n_h, n_y)    
  W1 = pars["W1"]
  b1 = pars["b1"]
  W2 = pars["W2"]
  b2 = pars["b2"]

  for i in range(0, num_iterations):
    A1, cache1 = linear_activation_fwd(I, W1, b1, "relu")
    A2, cache2 = linear_activation_fwd(A1, W2, b2, "sigmoid")
    cost = cost(A2, O)
    dA2 = - (np.divide(O, A2) - np.divide(1 - O, 1 - A2))
    dA1, dW2, db2 = linear_activation_bkward(dA2, cache2, "sigmoid")
    dA0, dW1, db1 = linear_activation_bkward(dA1, cache1, "relu")
    gradients['dW1'] = dW1
    gradients['db1'] = db1
    gradients['dW2'] = dW2
    gradients['db2'] = db2
    pars = update_pars(pars, gradients, learning_rate)
    W1 = pars["W1"]
    b1 = pars["b1"]
    W2 = pars["W2"]
    b2 = pars["b2"]
    if i % 200 == 0 or i == num_iterations:
      costs.append(cost)

  return pars, costs


def deep_model(I, O, layers_dims, learning_rate = 0.0075, num_iterations = 3000):    
    costs = []         

    pars = init_pars_deep(layers_dims)    
  
    for i in range(0, num_iterations):      
        AL, caches = L_model_fward(I, pars)     
        cost = cost(AL, O)     
        gradients = L_model_bkward(AL, O, caches)        
        pars = update_pars(pars, gradients, learning_rate)        
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    
    return pars, costs























