import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def initialize_parameters(layers_dims):
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = torch.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
        parameters['b' + str(l)] = torch.zeros((layers_dims[l], 1))
        
    return parameters


def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(1, L + 1):
        v['dW' + str(l)] = torch.zeros(parameters['W' + str(l)].shape)
        v['db' + str(l)] = torch.zeros(parameters['b' + str(l)].shape)
        s['dW' + str(l)] = torch.zeros(parameters['W' + str(l)].shape)
        s['db' + str(l)] = torch.zeros(parameters['b' + str(l)].shape)
    
    return v, s


def random_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[1]
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape(Y.shape[0], m)
    
    inc = mini_batch_size
    
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size * k : mini_batch_size * (k + 1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * k : mini_batch_size * (k + 1)]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches:]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches


def linear_activation_forward(A_prev, W, b, activation):
    W = W.to(torch.float64)
    A_prev = A_prev.to(torch.float64)
    Z = torch.mm(W, A_prev) + b
    
    if activation == 'relu':
        A = torch.relu(Z)
    elif activation == 'linear':
        A = Z
    
    cache = (A_prev, W, b, Z)
    return A, cache


def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='relu')
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='linear')
    caches.append(cache)
          
    return AL, caches


def compute_cost(AL, Y, parameters, lambd):
    m = Y.shape[1]
    L = len(parameters) // 2
    
    cross_entropy_cost = nn.functional.mse_loss(AL, Y)
    
    L2_regularization_cost = 0
    for l in range(1, L + 1):
        Wl = parameters["W" + str(l)]
        L2_regularization_cost += torch.sum(Wl**2)
    L2_regularization_cost = lambd * L2_regularization_cost / (2 * m)
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost


def linear_activation_backward(dA, cache, activation):
    A_prev, W, b, Z = cache
    
    if activation == 'relu':
        dZ = dA * (Z > 0).float()  # Convert boolean to float
    elif activation == 'linear':
        dZ = dA
    else:  # Assuming sigmoid activation
        dZ = dA * torch.sigmoid(Z) * (1 - torch.sigmoid(Z))
    
    m = A_prev.shape[1]
    dW = torch.mm(dZ, A_prev.t()) / m
    db = torch.sum(dZ, dim=1, keepdim=True) / m
    dA_prev = torch.mm(W.t(), dZ)
    
    return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.view(AL.shape)  # Y is the same shape as AL

    dAL = -2 * (Y - AL)
    
    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, 'linear')
    grads['dA' + str(L - 1)] = dA_prev_temp
    grads['dW' + str(L)] = dW_temp
    grads['db' + str(L)] = db_temp

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, 'relu')
        grads['dA' + str(l)] = dA_prev_temp
        grads['dW' + str(l + 1)] = dW_temp
        grads['db' + str(l + 1)] = db_temp

    return grads


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.0007,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L + 1):
        v['dW' + str(l)] = beta1 * v['dW' + str(l)] + (1 - beta1) * grads['dW' + str(l)]
        v['db' + str(l)] = beta1 * v['db' + str(l)] + (1 - beta1) * grads['db' + str(l)]

        v_corrected['dW' + str(l)] = v['dW' + str(l)] / (1 - beta1 ** t)
        v_corrected['db' + str(l)] = v['db' + str(l)] / (1 - beta1 ** t)

        s['dW' + str(l)] = beta2 * s['dW' + str(l)] + (1 - beta2) * grads['dW' + str(l)] ** 2
        s['db' + str(l)] = beta2 * s['db' + str(l)] + (1 - beta2) * grads['db' + str(l)] ** 2

        s_corrected['dW' + str(l)] = s['dW' + str(l)] / (1 - beta2 ** t)
        s_corrected['db' + str(l)] = s['db' + str(l)] / (1 - beta2 ** t)

        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * v_corrected['dW' + str(l)] / (torch.sqrt(s_corrected['dW' + str(l)]) + epsilon)
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * v_corrected['db' + str(l)] / (torch.sqrt(s_corrected['db' + str(l)]) + epsilon)

    return parameters, v, s, v_corrected, s_corrected


def update_lr(learning_rate0, epoch_num, decay_rate):
    learning_rate = 1 / (1 + decay_rate * epoch_num) * learning_rate0
    
    return learning_rate


def model(X, Y, layers_dims, learning_rate=0.0007, lambd=0, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=5000, print_cost=True, decay=None, decay_rate=1):

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    m = X.shape[1]                   # number of training examples
    lr_rates = []
    learning_rate0 = learning_rate   # the original learning rate
    
    parameters = initialize_parameters(layers_dims)
    v, s = initialize_adam(parameters)
    
    for i in range(num_epochs):
        minibatches = random_mini_batches(X, Y, mini_batch_size)
        cost_total = 0
        
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            a3, caches = forward_propagation(minibatch_X, parameters)

            cost_total += compute_cost(a3, minibatch_Y, parameters, lambd)

            grads = backward_propagation(a3, minibatch_Y, caches)

            t = t + 1  # Adam counter
            parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
                                                                  t, learning_rate, beta1, beta2, epsilon)
        cost_avg = cost_total / m
        
        if decay:
            learning_rate = update_lr(learning_rate0, i, decay_rate)
        
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)
                
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


def predict(parameters, X):
    A, _ = forward_propagation(X, parameters)
    
    return A
