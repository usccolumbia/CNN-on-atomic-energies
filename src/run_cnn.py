import theano
import theano.tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import matplotlib.pyplot as plt
import cnn

def TrainCNN(train_set_x,train_set_y,valid_set_x,valid_set_y,test_set_x,learning_rate,num_epochs,num_filters,mini_batch_size,reg):
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

    # First convolution and pooling layer
    [layer0_output, layer0_params] = cnn.convLayer(
        rng,
        data_input=layer0_input,
        image_spec=(mini_batch_size, 1, 80, 80),
        filter_spec=(num_filters[0], 1, 9, 9),
        pool_size=(2,2),
        activation=T.nnet.relu)

    # Second convolution and pooling layer
    [layer1_output, layer1_params] = cnn.convLayer(
        rng,
        data_input=layer0_output,
        image_spec=(mini_batch_size, num_filters[0], 36, 36),
        filter_spec=(num_filters[1],num_filters[0],5,5),
        pool_size=(2,2),
        activation=T.nnet.elu)
    
    [layer2_output, layer2_params] = cnn.convLayer(
        rng,
        data_input=layer1_output,
        image_spec=(mini_batch_size, num_filters[1], 16, 16),
        filter_spec=(num_filters[2], num_filters[1], 5, 5),
        pool_size=( 2,2),
        activation=T.nnet.elu)

    #[layer3_output, layer3_params] = cnn.convLayer(
    #    rng,
    #    data_input=layer2_output,
    #    image_spec=(mini_batch_size, num_filters[2], 6, 6),
    #    filter_spec=(num_filters[3], num_filters[2], 3, 3),
    #    pool_size=( 2,2),
    #    activation=T.tanh)


    # Flatten the output into dimensions:
    # mini_batch_size x 432
    fc_layer_input = layer2_output.flatten(2)

    # The fully connected layer operates on a matrix of
    # dimensions: mini_batch_size x 1098# It clasifies the values using the softmax function.
    #[fc1_layer_output, fc1_layer_params] = cnn.fullyConnectedLayer(
    #    rng=rng,
    #    data_input=fc_layer_input,
    #    num_in=num_filters[3]*2*2,
    #    num_out=10)

    [E_pred, fc_layer_params] = cnn.fullyConnectedLayer(
        rng=rng,
        data_input=fc_layer_input,
        num_in=num_filters[2]*6*6,
        num_out=1)
    
    # Cost that is minimised during stochastic descent. Includes regularization
    cost = cnn.MSE(y,E_pred)
    # Cost to be evaluated at kaggle, checked at validation
    #cost_val = cnn.RMSLE(y,E_pred)
    #

    L2_reg=T.mean(T.sqr(layer0_params[0]))
    L2_reg=L2_reg+T.mean(T.sqr(layer1_params[0]))
    L2_reg=L2_reg+T.mean(T.sqr(layer2_params[0]))
    L2_reg=L2_reg+T.mean(T.sqr(fc_layer_params[0]))
#    L2_reg=L2_reg+T.sum(T.sqr(layer2_params[0]))/(num_filters[1]*num_filters[0]*2*3)
#    L2_reg=L2_reg+T.sum(T.sqr(layer2_params[1])/num_filters[2])

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
    
    

    # List of parameters to be fit during training
    params = fc_layer_params + layer0_params + layer1_params + layer2_params#+ fc2_layer_params + layer3_params
    
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

    # Get convolution and fully connected layer weights as numpy arrays
    w0_conv = layer0_params[0].get_value()
    w1_conv = layer1_params[0].get_value()
    w2_conv = layer2_params[0].get_value()      
    w_fc1 = fc_layer_params[0].get_value()
    #w_fc2 = fc2_layer_params[0].get_value()

    # Save weights from only the bottom layer of the filter
    w0_conv = np.array(w0_conv[:, 0])
    w1_conv = np.array(w1_conv[:, 0])
    w2_conv = np.array(w2_conv[:, 0])
    
    # Create buffer arrays for the channel and fully connected
    # layer weights. From all channels, three somewhat arbitrary
    # weights are gathred.
    w0_arr1 = np.array([w0_conv[:, 1, 1]]);
    w0_arr2 = np.array([w0_conv[:, 2, 2]]);
    w0_arr3 = np.array([w0_conv[:, 3, 3]])

    w1_arr1 = np.array([w1_conv[:, 1, 1]]);
    w1_arr2 = np.array([w1_conv[:, 2, 2]]);
    w1_arr3 = np.array([w1_conv[:, 3, 3]])
    
    w2_arr1 = np.array([w2_conv[:, 1, 1]]);
    w2_arr2 = np.array([w2_conv[:, 2, 2]]);

    wfc1_arr1 = np.array([w_fc1[0]])
    wfc1_arr2 = np.array([w_fc1[1]])
    wfc1_arr3 = np.array([w_fc1[2]])

    #wfc2_arr1 = np.array([w_fc2[0]])
    #wfc2_arr2 = np.array([w_fc2[1]])
    #wfc2_arr3 = np.array([w_fc2[2]])
            
    
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
            # Save training error
            train_error.append(float(cost_ij))
            
            valid_losses = [valid_model(i) for i in range(n_valid_batches)]
            # Compute the mean prediction error across all the mini-batches.
            valid_score = np.mean(valid_losses)
            # Save validation error
            valid_error.append(valid_score)

            print("Iteration: "+str(iter+1)+"/"+str(num_epochs*n_train_batches)+", training error: "+str(cost_ij)+", validation error: "+str(valid_score))
            
            # Obtain the weights for visualisation
            w0_conv = layer0_params[0].get_value()
            wt0 = np.array(w0_conv[:, 0])
            w0_arr1 = np.append(w0_arr1, [wt0[:, 1, 1]], axis=0)
            w0_arr2 = np.append(w0_arr2, [wt0[:, 2, 2]], axis=0)
            w0_arr3 = np.append(w0_arr3, [wt0[:, 3, 3]], axis=0)
            
            w1_conv = layer1_params[0].get_value()
            wt1 = np.array(w1_conv[:, 0])
            w1_arr1 = np.append(w1_arr1, [wt1[:, 1, 1]], axis=0)
            w1_arr2 = np.append(w1_arr2, [wt1[:, 2, 2]], axis=0)
            w1_arr3 = np.append(w1_arr3, [wt1[:, 3, 3]], axis=0)
            
            w2_conv = layer2_params[0].get_value()
            wt2 = np.array(w2_conv[:, 0])
            w2_arr1 = np.append(w2_arr1, [wt2[:, 1, 1]], axis=0)
            w2_arr2 = np.append(w2_arr2, [wt2[:, 2, 2]], axis=0)

            w_fc1 = fc_layer_params[0].get_value()
            wfc1_arr1 = np.append(wfc1_arr1, [w_fc1[0]], axis=0)
            wfc1_arr2 = np.append(wfc1_arr2, [w_fc1[1]], axis=0)
            wfc1_arr3 = np.append(wfc1_arr3, [w_fc1[2]], axis=0)

            if (iter%20==0):
                # Get predicted energies from validation set
                E = np.zeros((n_valid_batches*mini_batch_size,1))
                step=0
                for i in range(n_valid_batches):
                    buf = predict(i)
                    for j in range(mini_batch_size):
                        E[step,0]=buf[j]
                        step=step+1
                np.savetxt('test/E_pred_'+str(iter)+'.txt',E)

    # Predict energies for test set
    E_test = np.zeros((n_test_batches*mini_batch_size,1))
    step=0
    for i in range(n_test_batches):
        buf = test_model(i)
        for j in range(mini_batch_size):
            E[step,0]=buf[j]
            step=step+1
    # Return values:
    # * train_error <list of floats>
    # * valid_error <list of floats>
    # * w1          <np.array((#iterations,#channels@layer1))>
    # * w2
    # * w3
    # * wfc         <np.array((#iterations,1))>
    return train_error, valid_error, w0_arr1,w0_arr2,w0_arr3,w1_arr1,w1_arr2,w1_arr3,w2_arr1,w2_arr2,wfc1_arr1,wfc1_arr2,wfc1_arr3,E_test#wfc2_arr1,wfc2_arr2,wfc2_arr3
