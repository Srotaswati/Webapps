import numpy as np
import h5py
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from getdata import * # load datasets
from mlutils import * # utility functions

plt.rcParams['figure.figsize'] = (10.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)

def neural_network(X, Y, layers_dims, optimizer="gd", num_epochs=3000, mini_batch_size=64, beta1=0.9, beta2=0.999, epsilon=1e-8, print_cost=False, learning_rate=0.0007, lambd=0, keep_prob=1):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = [] 
    t = 0
    seed = 10
    m = X.shape[1]
    
    # Parameters initialization. 
    parameters = initialize_parameters(layers_dims)
    
    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    
    for i in range(num_epochs):
        
        # Define the random minibatches. Increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost = 0
        
        for minibatch in minibatches:
            
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            
            # Forward propagation:       
            AL, caches = forward_prop(minibatch_X, parameters, keep_prob)
        
            # Compute cost.
            cost +=  compute_cost(AL, minibatch_Y, parameters, lambd)
    
            # Backward propagation.
            grads = back_prop(AL, minibatch_Y, caches, lambd, keep_prob)
 
            # Update parameters.
            if optimizer == "gd":
                parameters = update_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,  epsilon)
            ### END CODE HERE ###
                
            # Print the cost every 100 training example
        cost_avg = cost / m
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost_avg))
        if i % 100 == 0:
            costs.append(cost_avg)
            
    # plot the cost
    plt.subplot(1, 2, 1)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate =" + str(learning_rate))
    
    # Plot decision boundary
    plt.subplot(1, 2, 2)
    plt.title("Model with " + optimizer + " optimization")
    axes = plt.gca()
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T, keep_prob), X, Y)
    
    # print accuracy
    pred_train = predict(X, Y, parameters, keep_prob)
    
    result = {"costs" : costs,
         "Y_prediction_train" : pred_train, 
         "keep_prob" : keep_prob, 
         "reg_param" : lambd,
         "alpha" : learning_rate,
         "num_epochs": num_epochs,
         "parameters" : parameters,
         "gradients" : grads}
    
    return result
