import numpy as np
import theano
import theano.tensor as T
import random as rd
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
#import matplotlib.pyplot as plt


def gradient_updates_Adam(cost, params, learning_rate):
    # Function to return an update list for the parameters to be updated
    # cost: MSE cost Theano variable
    # params : parameters coming from hidden and output layers
    # learning rate: learning rate defined as hyperparameter
    # Outputs:
    # updates : updates to be made and to be defined in the train_model function.
    updates = []
    eps = 1e-4 # small constant used for numerical stabilization.
    beta1 = 0.9
    beta2 = 0.999
    # beta1 and beta2 are the exponential decay rates
    # for moment estimates, in [0,1).
    # suggested defaults: 0.9 and 0.999 respectively
    for param in params:
        t = theano.shared(1)
        s = theano.shared(param.get_value(borrow=True)*0.)
        r = theano.shared(param.get_value(borrow=True)*0.)
        s_new = beta1*s + (1.0-beta1)*T.grad(cost, param)
        r_new = beta2*r + (1.0-beta2)*(T.grad(cost, param)**2)
        updates.append((s, s_new))
        updates.append((r, r_new))
        s_hat = s_new/(1-beta1**t)
        r_hat = r_new/(1-beta2**t)
        updates.append((param, param - learning_rate*s_hat/(np.sqrt(r_hat)+eps) ))
    updates.append((t, t + 1))
    return updates

def pooling(input_pool, size, ignore_border=True):
    # Function to perform Max-Pooling on each feature map
    
    # Inputs:
    # input_pool - feature maps obtained as output from convolution layer.
    # size - specification of downsampling (pooling) factor.
    #        tuple format: (# of rows, # of columns)
    
    # Outputs:
    # pool_out - pooled output.
    #            dimensions: (# of channels, conv_output_height/#rows,
    #                         conv_output_width/#rows)
    
    pool_out = pool.pool_2d(input=input_pool, ws=size, ignore_border=ignore_border)
    return pool_out

def convLayer(rng, data_input, filter_spec, image_spec, pool_size, activation):
    # Function that defines the convolution layer. Calls the
    # activation function and then Pooling function.
    
    # Inputs:
    # rng - random number generator used to initialize weights.
    # data_input - symbolic input image tensor.
    # filter_spec - dimensions of filter in convolution layer.
    #               tuple format:(# of channels, depth, height, width)
    # image_spec - specifications of input images.
    #              tuple format:(batch size, color channels, height, width)
    # pool_size - specification of downsampling (pooling) factor.
    #             tuple format: (# of rows, # of columns)
    # activation - activation function to be used.
    
    # Outputs:
    # output - tensor containing activations fed into next layer.
    # params - list containing layer parameters
    
    # Weigths
    W = theano.shared(
        np.asarray(rng.normal(loc=0, scale=0.1, size=filter_spec)),
        borrow=True)
    # Bias is a 1 D tensor -- one bias per output feature map.
    # Initialised with zeros.
    b = theano.shared(np.zeros((filter_spec[0],)), borrow=True)
    
    # Convolve input with specifications. This is Theano's convolution
    # function. It takes as input the data tensor, filter weights, filter
    # specifications and the image specifications. 
    
    conv_op_out = conv2d(
        input=data_input,
        filters=W,
        filter_shape=filter_spec,
        input_shape=image_spec,
        border_mode='valid')
    
    # Add the bias term and use the specified activation function/
    # non-linearity.
    # b.dimshuffle returns a view of the bias tensor with permuted dimensions.
    layer_activation = activation(conv_op_out + b.dimshuffle('x', 0, 'x', 'x'))
    
    # Perform pooling on the activations.
    output = pooling(input_pool=layer_activation, size=pool_size)
    
    # Combine the weights and biases into a single list
    params = [W, b]
    return output, params

def fullyConnectedLayer(rng,data_input, num_in,num_out,activation):
    # Function to create the fully-connected layer and makes use of the
    # output from the previous layer. It is the final layer in the
    # convolutional neural network architecture and comprises of the
    # Softmax activations.
    
    # Inputs:
    # data_input - input for the softmax layer.
    #              Symbolic tensor of dimensions:
    #              (mini_batch_size, # channels * 12 * 12)
    # num_in - number of input units. Dimensions would be:
    #           (# channels * 12 * 12)
        
    # Outputs:
    # p_y_given_x - class-membership probabilities.
    # y_pred - class with maximal probability
    # params - parameters of the layer
    
    # Creating a shared variable for weights that are initialised with samples
    # drawn from a gaussian distribution with 0 mean and standard deviation of
    # 0.1. This is just a random initialisation.
    w_bound=np.sqrt(6./(num_in+num_out))
    W = theano.shared(
        value=np.asarray(
            rng.uniform(low=-w_bound,
                        high=w_bound,
                        size=(num_in,num_out))),
        name='W',
        borrow=True)
    
    # Creating a shared variable for biases that are initialised with
    # zeros.
    b = theano.shared(
        value=np.zeros((num_out,)),
        name='b',
        borrow=True)
    
    # Compute predicted energies
    
    E_pred = activation(T.dot(data_input, W) + b)
        
    # Combine weights and biases into a single list.
    params = [W, b]
    return E_pred, params

def MSE(y,y_pred):
    # Function to compute root mean square error
    cost_MSE=T.mean(T.sqr(y-y_pred))
    return cost_MSE
    
def RMSLE(y, y_pred):
    # Function to compute the cost that is to be minimised.

    # Inputs:
    # y      : expected energy
    # y_pred : calculated energy
    # msb    : minibatch size
    
    # Outputs:
    # cost_RMSLE - the computed mean square logarithmic error

    #cost_RMSLE = T.sqrt(1/msb*T.sum(T.sqr(T.log(y_pred+1)-T.log(y+1))))

    cost_RMSLE=T.sqrt(T.mean(T.sqr(T.log(y_pred+1)-T.log(y+1))))
    
    return cost_RMSLE

def negative_log_lik(p_y_given_x,y):
    # CURRENTLY NOT WORKING!!!

    # Function to compute the cost that is to be minimised.
    # Here, we compute the negative log-likelihood.

    
    
    # Inputs:
    # y - expected class label
    # p_y_given_x - class-membership probabilities
    
    # Outputs:
    # cost_log - the computed negative log-lik cost

    yint=T.argmax(y,axis=1)
    
    # Generate the relevant row indices
    rows = T.arange(yint.shape[0])
    
    # Generate the relevant column indices
    cols = yint
    
    # Computing the log probabilities
    log_prob = T.log(p_y_given_x)
    
    # Obtain the mean of the relevant entries. Loss is formally
    # defined over the sum of the individual error terms as in
    # the equation above. However, we use mean instead to speed
    # up convergence.
    cost_log = -T.sum(log_prob[rows, cols])
    return cost_log

def categorical_cross_entropy(y_pred,y):
    '''
    Function to compute the negative cross-entropy cost
    Between approximated distribution and a true distribution

    Inputs:
    2D matrices describing true and predicted likelihoods
    
    Output:
    0-D value
    '''

    return T.mean(T.nnet.categorical_crossentropy(y_pred,y))
    

def class_errors(y_pred,y):
    # Function to compute to the number of wrongly classified
    # instances.
    
    # Inputs:
    # y - expected class label in onehot-format
    # y_pred - predicted class label
    
    # Outputs:
    # count_error - number of wrongly classified instances

    labels=T.argmax(y,axis=1)
    preds=T.argmax(y_pred,axis=1)
    
    count_error = T.mean(T.neq(preds, labels))
    return count_error


def linear_activation(x):
    return x



        
