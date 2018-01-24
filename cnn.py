import numpy as np
import theano
import theano.tensor as T
import random as rd
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
#import matplotlib.pyplot as plt
print('***** Import complete *****')

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

def pooling(input_pool, size):
    # Function to perform Max-Pooling on each feature map
    
    # Inputs:
    # input_pool - feature maps obtained as output from convolution layer.
    # size - specification of downsampling (pooling) factor.
    #        tuple format: (# of rows, # of columns)
    
    # Outputs:
    # pool_out - pooled output.
    #            dimensions: (# of channels, conv_output_height/#rows,
    #                         conv_output_width/#rows)
    
    pool_out = pool.pool_2d(input=input_pool, ws=size, ignore_border=True)
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
    
    # Creating a shared variable for weights that are initialised with samples
    # drawn from a gaussian distribution with 0 mean and standard deviation of
    # 0.1. This is just a random initialisation.
    m=image_spec[1]*image_spec[2]*image_spec[3]
    n=m*filter_spec[0]
    w_bound=np.sqrt(6./(m+n))
    W = theano.shared(np.asarray(rng.uniform(low=-w_bound,
                                             high=w_bound,
                                             size=filter_spec)), borrow=True)
    # Bias is a 1 D tensor -- one bias per output feature map.
    # Initialised with zeros.
    b = theano.shared(np.zeros((filter_spec[0],)), borrow=True)
    
    # Convolve input with specifications. This is Theano's convolution
    # function. It takes as input the data tensor, filter weights, filter
    # specifications and the image specifications. In our example, the
    # dimensions of the output of this operation would be:
    # mini_batch_size x 20 x 80 x 80
    
    conv_op_out = conv2d(
        input=data_input,
        filters=W,
        filter_shape=filter_spec,
        input_shape=image_spec)
    
    # Add the bias term and use the specified activation function/
    # non-linearity.
    # b.dimshuffle returns a view of the bias tensor with permuted dimensions.
    # In this case our bias tensor is originally of the dimension 9 x 1. The
    # dimshuffle operation used below, broadcasts this into a tensor of
    # 1 x 9 x 1 x 1. Note that there is one bias per output feature map.
    layer_activation = activation(conv_op_out + b.dimshuffle('x', 0, 'x', 'x'))
    
    # Perform pooling on the activations. It is required to reduce the spatial
    # size of the representation to reduce the number of parameters and
    # computation in the network. Hence, it helps to control overfitting
    # Output dimensions: (# channels, image height-filter height+1,
    #                     image width - filter width+1)
    # In our demo, the dimensions would be of mini_batch_size x 9 x 12 x 12
    output = pooling(input_pool=layer_activation, size=pool_size)
    
    # Combine the weights and biases into a single list
    params = [W, b]
    return output, params

def fullyConnectedLayer(rng,data_input, num_in):
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
    w_bound=np.sqrt(6./(num_in+1))
    W = theano.shared(
        value=np.asarray(
            rng.uniform(low=-w_bound,
                        high=w_bound,
                        size=(num_in,1))),
        name='W',
        borrow=True)
    
    # Creating a shared variable for biases that are initialised with
    # zeros.
    b = theano.shared(
        value=np.zeros((1,)),
        name='b',
        borrow=True)
    
    # Compute predicted energies
    E_pred = T.nnet.sigmoid(T.dot(data_input, W) + b)
        
    # Combine weights and biases into a single list.
    params = [W, b]
    return E_pred, params

def RMSLE(y, y_pred,msb):
    # Function to compute the cost that is to be minimised.

    # Inputs:
    # y      : expected energy
    # y_pred : calculated energy
    # msb    : minibatch size
    
    # Outputs:
    # cost_RMSLE - the computed mean square logarithmic error
    
    cost_RMSLE = T.sqrt(1/msb*T.sum((T.log(T.transpose(y_pred)+1)-T.log(y+1))**2))

    return cost_MSE








        
