import theano
import theano.tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cnn

def TrainCNN(train_set_x,train_set_y,valid_set_x,valid_set_y,learning_rate,num_epochs,num_filters,mini_batch_size,reg):
    # Seeding the random number generator
    rng = np.random.RandomState(23455)
    
    # Computing number of mini-batches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= mini_batch_size
    n_valid_batches //= mini_batch_size
    n_test_batches =  test_set_x.get_value(borrow=True).shape[0]
    n_test_batches //= mini_batch_size
    
    print('train: %d batches, validation: %d batches'
          % (n_train_batches, n_valid_batches))

    # mini-batch index
    mb_index = T.lscalar()
    # Song feature vectors ( mini_batch_size x 264 matrix)
    x = T.matrix('x')
    # image labels (264 D vector)
    y = T.ivector('y')

    print('***** Constructing model ***** ')
    
    # Reshaping matrix of mini_batch_size set of images into a
    # 4-D tensor of dimensions: mini_batch_size x 3 x 12 x 8
    # REMEMBER: pad input data to 288-dimensional if 3d input layers used
    layer0_input = x.reshape((mini_batch_size,1,6,4,12))
    
    # First convolution and pooling layer
    # 4D output tensor is of shape:
    #     mini_batch_size x 1 x 9 x 8 x 6
    # OR  mini_batch_size x 9 x 1 x 8 x 6
    [layer0_output, layer0_params] = cnn.convLayer(
        rng_np,
        data_input=layer0_input,
        image_spec=(mini_batch_size, 1, 6, 4, 12),
        filter_spec=(num_filters[0], 1, 4, 2, 3),
        pool_size=(2,1,2),
        activation=T.tanh,
        border_mode=(2,0,0) )
    
    [layer1_output, layer1_params] = cnn.convLayer(
        rng_np,
        data_input=layer0_output,
        image_spec=(mini_batch_size, num_filters[0], 4, 3, 5),
        filter_spec=(num_filters[1],num_filters[0],3,3,3),
        pool_size=(2,1,3),
        activation=T.tanh,
        border_mode=(1,0,0))
    
    #[layer2_output, layer2_params] = cnn.convLayer(
    #    rng_np,
    #    data_input=layer1_output,
    #    image_spec=(mini_batch_size, num_filters[1], 1, 3, 6),
    #    filter_spec=(num_filters[2], num_filters[1], 1, 2, 3),
    #    pool_size=( 1, 1,2),
    #    activation=T.tanh,
    #    border_mode='valid')


    # Flatten the output into dimensions:
    # mini_batch_size x 432
    fc_layer_input = layer1_output.flatten(2)

    # The fully connected layer operates on a matrix of
    # dimensions: mini_batch_size x 1098# It clasifies the values using the softmax function.
    [p_y_given_x, y_pred, fc_layer_params] = cnn.fullyConnectedLayer(
        rng=rng,
        data_input=fc_layer_input,
        num_in=num_filters[1]*2*1*1,
        num_out=10)
    
    # Cost that is minimised during stochastic descent. Includes regularization
    cost = cnn.negative_log_lik(y=y, p_y_given_x=p_y_given_x)
    L2_reg=T.sum(T.sqr(layer0_params[0]))/(num_filters[0]*5*5)
    L2_reg=L2_reg+T.sum(T.sqr(layer0_params[1]))/(num_filters[0])
    L2_reg=L2_reg+T.sum(T.sqr(fc_layer_params[0]))/(num_filters[1]*2*2*10)
    L2_reg=L2_reg+T.sum(T.sqr(fc_layer_params[1]))/10
    L2_reg=L2_reg+T.sum(T.sqr(layer1_params[0]))/(num_filters[1]*num_filters[0]*3*3)
    L2_reg=L2_reg+T.sum(T.sqr(layer1_params[1]))/num_filters[1]
    #L2_reg=L2_reg+T.sum(T.sqr(layer2_params[0]))/(num_filters[1]*num_filters[0]*2*3)
    #L2_reg=L2_reg+T.sum(T.sqr(layer2_params[1])/num_filters[2])

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
        cnn.errors(y, y_pred),
        givens={
            x: valid_set_x[
                mb_index * mini_batch_size:
                (mb_index + 1) * mini_batch_size
            ],
            y: valid_set_y[
                mb_index * mini_batch_size:
                (mb_index + 1) * mini_batch_size
            ]})
    
    #test_model = theano.function(
    #    [mb_index],
    #    y_pred,
    #    givens={
    #        x: test_set_x[
    #            0:mb_index
    #        ]})
    

    # List of parameters to be fit during training
    params = fc_layer_params + layer0_params + layer1_params #+ layer2_params
    
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
            
            if (iter%100==0):
                print('Iteration: '+str(iter)+', cost: '+str(cost_ij)+', epoch: '+str(epoch)+', accuracy: '+str(valid_score))
        # Compute the prediction error on each validation mini-batch by
        # calling the previously defined Theano function for validation.
        valid_losses = [valid_model(i) for i in range(n_valid_batches)]
        
        # Compute the mean prediction error across all the mini-batches.
        valid_score = np.mean(valid_losses)
                
    print('***** Training Complete *****')
    print('Validation error: '+str(valid_score))
    print('Final cost: '+str(cost_ij))
            
    #predictions=test_model(6542)
    #file = open('solution.csv','w')
    #file.write('Sample_id,Sample_label')
    #for i in range(Xtest.shape[0]):
    #    file.write(str(i)+str(predictions[i]))
