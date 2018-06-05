import six.moves.cPickle as pickle
import gzip
import os
import random as rd
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import cnn
import load_data

def test_MSE():
    A=np.zeros((2,1))
    B=np.zeros((2,1))
    A[0,0]=1
    A[1,0]=1

    y,y_pred,yy=load_data.shared_dataset(A,B,sample_size=2)
    cost=cnn.MSE(y,y_pred)
#    assert (cost < 1.0001)
#    assert (cost > 0.9999)
    print(cost.eval())

test_MSE()

def negative_log_lik(y, p_y_given_x):
    # Function to compute the cost that is to be minimised. 
    # Here, we compute the negative log-likelihood.
    
    # Inputs:
    # y - expected class label
    # p_y_given_x - class-membership probabilities
    
    # Outputs:
    # cost_log - the computed negative log-lik cost
    
    # Generate the relevant row indices
    rows = T.arange(y.shape[0])

    # Generate the relevant column indices
    cols = y;

    # Computing the log probabilities
    log_prob = T.log(p_y_given_x)
    
    # Obtain the mean of the relevant entries. Loss is formally
    # defined over the sum of the individual error terms as in
    # the equation above. However, we use mean instead to speed
    # up convergence.
    cost_log = -T.mean(log_prob[rows, cols])
    return cost_log

def errors(y, y_pred):
    # Function to compute the fraction of wrongly classified
    # instances.
    
    # Inputs:
    # y - expected class label
    # y_pred - predicted class label
    
    # Outputs:
    # count_error - number of wrongly classified instances
    
    # Counting the number of number of wrong predictions. T.neq
    # function returns 1 if the variables compared are not equal.
    # The mean would return the fraction of mismatches.
    count_error = T.mean(T.neq(y_pred, y))
    return count_error

def visualize_MISTtraining():
    '''
    A function to demonstrate how convolutional and fully 
    connected layers are used to train CNN to learn to label MNIST
    digits. 

    Same function is used in testing, without any output. 

    Downloads data from online, if mnist zip file is dot present.

    More plotting features and such should be included, now only
    terminal output.

    Benchmark error on test set with current settings:  0.0445
    '''
    dataset = 'mnist.pkl.gz'
    data_dir, data_file = os.path.split(dataset)
    rd.seed(23455)
    # Check if data file present
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join('', dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    # Download the file from MILA if not present 
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('***** Loading data *****')
    # Open the file
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    
    train_x,train_y=train_set
    valid_x,valid_y=valid_set
    test_x,test_y=test_set
    
    # Load data into tensors
    train_size = 6000
    test_set_x, test_set_y_float, test_set = load_data.shared_dataset(
        test_x,test_y,
        sample_size=train_size//3
        )
    valid_set_x, valid_set_y_float, valid_set = load_data.shared_dataset(
        valid_x,valid_y,
        sample_size=train_size//3
        )
    train_set_x, train_set_y_float, train_set = load_data.shared_dataset(
        train_x,train_y,
        sample_size=train_size
        )

    train_set_y=T.cast(train_set_y_float,'int32')
    valid_set_y=T.cast(valid_set_y_float,'int32')
    test_set_y=T.cast(test_set_y_float,'int32')
    
    # Training set dimension: 6000 x 784
    print('Training set: %d samples'
          %(train_set_x.get_value(borrow=True).shape[0])) 
    # Test set dimension: 2000 x 784
    print('Test set: %d samples'
          %(test_set_x.get_value(borrow=True).shape[0]))
    # Validation set dimension: 2000 x 784
    print('Validation set: %d samples'
          %(valid_set_x.get_value(borrow=True).shape[0]))
    print('The training set looks like this: ')
    print(train_set[0])
    print('The labels looks like this:')
    print(train_set[1])
    
    # set learning rate used for Stochastic Gradient Descent
    learning_rate = 0.005
    # set number of training epochs
    num_epochs = 4
    # set number of kernels for each convolution layer
    # for e.g. 2 layers - [20, 50]. layer1 = 20, layer2 = 50
    num_filters = [9]
    # set mini-batch size to be used
    mini_batch_size = 50

    # Seeding the random number generator
    rng = np.random.RandomState(23455)

    # Computing number of mini-batches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= mini_batch_size
    n_valid_batches //= mini_batch_size
    n_test_batches //= mini_batch_size
    
    
    print('train: %d batches, test: %d batches, validation: %d batches'
          % (n_train_batches, n_test_batches, n_valid_batches))
    
    # mini-batch index
    mb_index = T.lscalar()
    # rasterised images
    x = T.matrix('x')
    # image labels
    y = T.ivector('y')

    print('***** Constructing model ***** ')

    # Reshaping matrix of mini_batch_size set of images into a 
    # 4-D tensor of dimensions: mini_batch_size x 1 x 28 x 28
    layer0_input = x.reshape((mini_batch_size, 1, 28, 28))
    
    # First convolution and pooling layer
    # 4D output tensor is of shape:
    # mini_batch_size x 9 x 12 x 12
    [layer0_output, layer0_params] = cnn.convLayer(
        rng,
        data_input=layer0_input,
        image_spec=(mini_batch_size, 1, 28, 28),
        filter_spec=(num_filters[0], 1, 5, 5),
        pool_size=(2, 2),
        activation=T.tanh)

    # Flatten the output into dimensions:
    # mini_batch_size x 1296
    fc_layer_input = layer0_output.flatten(2)
    
    # The fully connected layer operates on a matrix of 
    # dimensions: mini_batch_size x 1296
    # It clasifies the values using the softmax function.
    [y_lin, fc_layer_params] = cnn.fullyConnectedLayer(
        rng,
        data_input=fc_layer_input,
        num_in=num_filters[0]*12*12,
        num_out=10)
    
    # The likelihood of the categories
    p_y_given_x = T.nnet.softmax(y_lin)
    # Predictions
    y_pred =  T.argmax(p_y_given_x,axis=1)

    # Cost that is minimised during stochastic descent. 
    cost = negative_log_lik(y=y, p_y_given_x=p_y_given_x)
    
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
        errors(y, y_pred),
        givens={
            x: valid_set_x[
                mb_index * mini_batch_size:
                    (mb_index + 1) * mini_batch_size
                ],
            y: valid_set_y[
                mb_index * mini_batch_size:
                    (mb_index + 1) * mini_batch_size
                ]})
    
    # Create a Theano function that computes the mistakes on the test set.
    # This evaluated our model's accuracy.
    test_model = theano.function(
        [mb_index],
        errors(y, y_pred),
        givens={
            x: test_set_x[
                mb_index * mini_batch_size:
                    (mb_index + 1) * mini_batch_size
                ],
            y: test_set_y[
                mb_index * mini_batch_size:
                    (mb_index + 1) * mini_batch_size
                ]})

    # List of parameters to be fit during training
    params = fc_layer_params + layer0_params
    # Creates a list of gradients
    grads = T.grad(cost, params)

    # Creates a function that updates the model parameters by SGD.
    # The updates list is created by looping over all 
    # (params[i], grads[i]) pairs.
    #updates = [(param_i, param_i - learning_rate * grad_i)
    #           for param_i, grad_i in zip(params, grads)]

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
    
    # Some code to help with the plotting. 
    # You don't need to go through the plotting 
    # code in detail.
    iter = 0
    epoch = 0
    cost_ij = 0

    train_costs=[]
    valid_accuracy=[]
    # This is where we call the previously defined Theano functions. 
    print('***** Training model *****')
    while (epoch < num_epochs):
        print('epoch: '+str(epoch))
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            # Compute number of iterations performed or total number
            # of mini-batches executed.
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            # Perform the training of our convolution neural network.
            # Obtain the cost of each minibatch specified using the 
            # minibatch_index.
            cost_ij = train_model(minibatch_index)
            print('iter: '+str(iter)+', cost_ij: '+str(cost_ij))
            train_costs.append(cost_ij)
        # Compute the prediction error on each validation mini-batch by
        # calling the previously defined Theano 

        valid_losses = [valid_model(i) for i in range(n_valid_batches)]
    
        # Compute the mean prediction error across all the mini-batches.
        valid_score = np.mean(valid_losses)
        valid_accuracy.append(valid_score)
        
    print('***** Training Complete *****')

    test_losses = [test_model(i) for i in range(n_test_batches)]
    # Compute the mean prediction error across all the mini-batches.                                               
    test_score = np.mean(test_losses)

    print('Accuracy on the test set: '+str(test_score))

    fig,(ax1,ax2)=plt.subplots(2)
    ax1.plot(train_costs)
    ax2.plot(valid_accuracy)
    plt.show()

##visualize_MISTtraining()
    
def visualize_sinetraining(Npoints,Nnodes,Nsteps,learning_rate):
    
    pi=3.14159265358

    xtrain=np.linspace(0,7,Npoints)
    ytrain=np.sin(xtrain)
    
    Xtrain=np.zeros((Npoints,1))
    for i in range(Npoints):
        Xtrain[i]=xtrain[i]

    Ytrain=np.sin(Xtrain)


    rng = np.random.RandomState(23455)
    
    x=T.matrix('x')
    y=T.matrix('y')
    
    [hout, params_1] = cnn.fullyConnectedLayer(
        rng=rng,
        data_input=x,
        num_in=1,
        num_out=Nnodes)
    
    [y_pred_lin, params_2] = cnn.fullyConnectedLayer(
        rng=rng,
        data_input=T.tanh(hout),
        num_in=Nnodes,
        num_out=1)
    y_pred=T.tanh(y_pred_lin)
    cost=cnn.MSE(y,y_pred)
    
    params = params_1 + params_2
    
    updates = cnn.gradient_updates_Adam(cost,params,learning_rate)
    
    train = theano.function(
        inputs=[x,y],
        outputs=[y_pred,cost,hout],
        updates=updates)
    
    plt.ion()
    fig=plt.figure()
    ax1=fig.add_subplot(311)
    ax2=fig.add_subplot(312)
    ax3=fig.add_subplot(313)
    ax1.plot(xtrain,ytrain)
    
#plt.ion()
    
    ax1.plot(xtrain,ytrain,'b-')
    errors=[]
    for i in range(Nsteps):
        pred,cost_i,hout=train(Xtrain,Ytrain)
        if(i%20==0):
            ax1.clear()
            line11,=ax1.plot(xtrain,ytrain,'b--', label='Inline label')
            line12,=ax1.plot(Xtrain,pred,'r-',  label='Inline label')
            line11.set_label('Training data')
            line12.set_label('prediction')
            ax1.legend() 
            ax1.set_title('Prediction')
            errors.append(cost_i)
            ax3.clear()
            line31,=ax3.plot(errors,'r-+', label='Inline label')
            ax3.set_title('Error')
            line31.set_label(str(cost_i))
            ax3.legend()
            ax2.clear()
            houtbout=T.tanh(hout)
            ax2.plot(Xtrain,houtbout)
            ax2.set_title('Activations')
            fig.canvas.draw()
            plt.pause(0.05)
    fig.canvas.draw()
    print('Final error: '+str(cost_i))



