import theano
import theano.tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import matplotlib.pyplot as plt
import random as rd
import cnn
import hyppar
import datapar
import load_data
import statistics

def CNNStructure(input,mini_batch_size,rng):
    
    # Use: hyppar module
    Nchannel     = hyppar.Nchannel
    NCL          = hyppar.NCL
    NFC          = hyppar.NFC
    pool         = hyppar.pool
    filter       = hyppar.filter
    activation   = hyppar.activation
    image_spec_x = hyppar.image_spec_x
    image_spec_y = hyppar.image_spec_y

    activation_function = []
    for i in range(NCL):
        if (activation[i]=="tanh"):
            activation_function.append(T.tanh)
        elif (activation[i]=="relu"):
            activation_function.append(T.nnet.relu)
        elif (activation[i]=="elu"):
            activation_function.append(T.nnet.elu)
        elif (activation[i]=="sigmoid"):
            activation_function.append(T.nnet.sigmoid)
        else :
            print(str(i+1)+": UNKNOWN ACTIVATION!!!!!!!!")

    # THIS IS NOT YET IMPLEMENTED TO FCL
    if (hyppar.fc_activation=="tanh"):
        fc_activation=T.tanh
    elif (hyppar.fc_activation=="relu"):
        fc_activation=T.nnet.relu
    elif (hyppar.fc_activation=="elu"):
        fc_activation=T.nnet.elu
    elif (hyppar.fc_activation=="sigmoid"):
        fc_activation=T.nnet.sigmoid
    #else :
    #    print("FC: UNKNOWN ACTIVATION!!!!!!!!")

    cn_output = []
    params    = []
    # This is the loop of the convolutions
    for i in range(NCL):
        [layer_output, layer_params] = cnn.convLayer(
            rng,
            data_input=input,
            image_spec=(mini_batch_size, 1, image_spec_x[i], image_spec_y[i]),
            filter_spec=(Nchannel[i+1], Nchannel[i], filter[i][0], filter[i][1]),
            pool_size=(pool[i][0],pool[i][1]),
            activation=activation_function[i])

        cn_output = cn_output + [layer_output]
        params = params + layer_params
        input = layer_output

    fc_layer_input = layer_output.flatten(2)

    # The fully connected layer operates on a matrix of                                                          
    [E_pred, fc_layer_params] = cnn.fullyConnectedLayer(
        rng=rng,
        data_input=fc_layer_input,
        num_in=Nchannel[NCL]*image_spec_x[-1]*image_spec_y[-1],
        num_out=1)

    params = params + fc_layer_params

    return E_pred, cn_output, params
    

def TrainCNN():
    
    # Training, validation and test data
    valid_set_x, valid_set_y, valid_set = load_data.shared_dataset(
        datapar.Xval, datapar.Yval,
        sample_size=hyppar.Nval)
    train_set_x, train_set_y, train_set = load_data.shared_dataset(
        datapar.Xtrain, datapar.Ytrain,
        sample_size=hyppar.Ntrain)
    test_set_x = load_data.shared_testset(datapar.Xtest)

    # Hyperparameters
    learning_rate   = hyppar.learning_rate
    num_epochs      = hyppar.Nepoch
    num_filters     = hyppar.Nchannel
    mini_batch_size = hyppar.mbs
    reg             = hyppar.reg

    # Random set for following activations
    rset = rd.sample(range(valid_set_x.get_value(borrow=True).shape[0]),mini_batch_size)
    print(rset)
    # Seeding the random number generator
    rng = np.random.RandomState(23455)
    
    # Computing number of mini-batches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= mini_batch_size
    n_valid_batches //= mini_batch_size
    n_test_batches //= mini_batch_size
        
    print('train: %d batches, validation: %d batches'
          % (n_train_batches, n_valid_batches))

    # mini-batch index
    mb_index = T.lscalar()
    # Coulomb matrices ( mini_batch_size x 80 x 80 matrix)
    x = T.matrix('x')
    # Target energies (1 x mini_batch_size)
    y = T.matrix('y')

    print('***** Constructing model ***** ')
    
    # Reshaping tensor of mini_batch_size set of images into a
    # 4-D tensor of dimensions: mini_batch_size x 1 x 80 x 80
    layer0_input = x.reshape((mini_batch_size,1,80,80))

    # Define the CNN function
    E_pred,cn_output,params=CNNStructure(layer0_input,mini_batch_size,rng)
    
    # Cost that is minimised during stochastic descent. Includes regularization
    cost = cnn.MSE(y,E_pred)

    L2_reg=0
    for i in range(len(params)):
        L2_reg=L2_reg+T.mean(T.sqr(params[i][0]))

    cost=cost+reg*L2_reg
    
    # Creates a Theano function that computes the mistakes on the validation set.
    # This performs validation.
    
    # Note: the givens parameter allows us to separate the description of the
    # Theano model from the exact definition of the inputs variable. The 'key'
    # that is passed to the graph is subsituted with the data from the givens
    # parameter. In this demo we built the model with a regular Theano tensor
    # and we use givens to speed up the GPU. We swap the input index with a
    # slice corresponding to the mini-batch of the dataset to use.
    
    # mb_index is the mini_batch_index
    valid_model = theano.function(
        [mb_index],
        cost,
        givens={
            x: valid_set_x[
                mb_index * mini_batch_size:
                (mb_index + 1) * mini_batch_size
            ],
            y: valid_set_y[
                mb_index * mini_batch_size:
                (mb_index + 1) * mini_batch_size
            ]})
    
    predict = theano.function(
        [mb_index],
        E_pred,
        givens={
            x : valid_set_x[
                mb_index * mini_batch_size:
                (mb_index+1) * mini_batch_size
                
            ]})

    test_model = theano.function(
        [mb_index],
        E_pred,
        givens={
            x: test_set_x[
                mb_index * mini_batch_size:
                (mb_index + 1) * mini_batch_size
            ]})
    
    get_activations = theano.function(
        [],
        cn_output,
        givens={x: valid_set_x[rset]})
    

    # Creates a function that updates the model parameters by SGD.
    # The updates list is created by looping over all
    # (params[i], grads[i]) pairs.
    updates = cnn.gradient_updates_Adam(cost,params,learning_rate)
    
    # Create a Theano function to train our convolutional neural network.
    train_model = theano.function(
        [mb_index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[
                mb_index * mini_batch_size:
                (mb_index + 1) * mini_batch_size
            ],
            y: train_set_y[
                mb_index * mini_batch_size:
                (mb_index + 1) * mini_batch_size
            ]})

        
    iter = 0
    epoch = 0
    cost_ij = 0
    valid_losses = [valid_model(i) for i in range(n_valid_batches)]
    valid_score = np.mean(valid_losses)

    train_error = []
    valid_error= []

    statistics.saveParameters(params)

     # This is where we call the previously defined Theano functions.
    print('***** Training model *****')
    while (epoch < num_epochs):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            # Compute number of iterations performed or total number
            # of mini-batches executed.
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            # Perform the training of our convolution neural network.
            # Obtain the cost of each minibatch specified using the
            # minibatch_index.
            cost_ij = train_model(minibatch_index)
            
            if iter%10==0:
                statistics.saveParameters(params)
            if iter%2==0:
                activations=get_activations()
                statistics.saveActivations(activations)
            
            # Save training error
            train_error.append(float(cost_ij))
            
            valid_losses = [valid_model(i) for i in range(n_valid_batches)]
            # Compute the mean prediction error across all the mini-batches.
            valid_score = np.mean(valid_losses)
            # Save validation error
            valid_error.append(valid_score)

            print("Iteration: "+str(iter+1)+"/"+str(num_epochs*n_train_batches)+", training error: "+str(cost_ij)+", validation error: "+str(valid_score))
            
            if (iter%20==0):
                # Get predicted energies from validation set
                E = np.zeros((n_valid_batches*mini_batch_size,1))
                step=0
                for i in range(n_valid_batches):
                    buf = predict(i)
                    for j in range(mini_batch_size):
                        E[step,0]=buf[j]
                        step=step+1
                np.savetxt('output/E_pred_'+str(iter)+'.txt',E)

    # Predict energies for test set
    E_test = np.zeros((n_test_batches*mini_batch_size,1))
    step=0
    for i in range(n_test_batches):
        buf = test_model(i)
        for j in range(mini_batch_size):
            E_test[step,0]=buf[j]
            step=step+1

    statistics.writeActivations()
    # Return values:
    statistics.saveParameters(params)
    statistics.writeParameters()

 
