import theano
import theano.tensor as T
import numpy as np
import six.moves.cPickle as pickle
import gzip
import os
import random as rd
import load_data
import cnn

def test_RMSLE():
    A=np.ones((2,1))
    B=np.ones((2,1))
    A[0,0]=2
    A[1,0]=2

    y,y_pred,yy=load_data.shared_dataset(A,B,sample_size=2)
    cost=cnn.RMSLE(y,y_pred)
    assert (cost.eval() - np.sqrt(( np.log(2) - np.log(3) )**2 ) < 0.001 )
    

def test_MSE():
    A=np.zeros((2,1))
    B=np.zeros((2,1))
    A[0,0]=1
    A[1,0]=1

    y,y_pred,yy=load_data.shared_dataset(A,B,sample_size=2)
    cost=cnn.MSE(y,y_pred)
    assert (cost.eval() < 1.0001)
    assert (cost.eval() > 0.9999)
    

def negative_log_lik(y, p_y_given_x):
    # Called by test_convLayer
    rows = T.arange(y.shape[0])
    cols = y;
    log_prob = T.log(p_y_given_x)
    cost_log = -T.mean(log_prob[rows, cols])
    return cost_log

def errors(y, y_pred):
    # Called by test_convLayer 
    count_error = T.mean(T.neq(y_pred, y))
    return count_error


def test_convLayer():
    '''
    Constructs a CNN with one convolutional and one fully connected layer.
    Then the function trains the network to interpret MNIST digits. Same 
    script with some output and plotting features is found from "test.py".
    
    Calls two functions for cost and accuracy from above.

    Test: digit labeling accuracy > 92%

    NOTE: Valid set is not present here.
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
    test_x,test_y=test_set

    # Load data into tensors                                                                                     
    train_size = 6000
    test_set_x, test_set_y_float, test_set = load_data.shared_dataset(
        test_x,test_y,
        sample_size=train_size/3
        )
    train_set_x, train_set_y_float, train_set = load_data.shared_dataset(
        train_x,train_y,
        sample_size=train_size
        )

    train_set_y=T.cast(train_set_y_float,'int32')
    test_set_y=T.cast(test_set_y_float,'int32')

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
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= mini_batch_size
    n_test_batches //= mini_batch_size

    # mini-batch index                                                                                           
    mb_index = T.lscalar()
    # rasterised images                                                                                          
    x = T.matrix('x')
    # image labels                                                                                               
    y = T.ivector('y')

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

    # This is where we call the previously defined Theano functions.                                             
    while (epoch < num_epochs):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            cost_ij = train_model(minibatch_index)

    test_losses = [test_model(i) for i in range(n_test_batches)]
    test_score = np.mean(test_losses)
    assert test_score < 0.08


def test_fullyConnectedLayer():
    '''
    Test that the fully connected layer works. This trains sine function 
    for a FCNN with one hidden layer of 4 units. For visualization check test.py.
    NOTE: Activations are done out of FC layer, since for atomic calculations
          linear activation is used.
    '''
    pi=3.14159265358

    xtrain=np.linspace(0,7,300)
    ytrain=np.sin(xtrain)

    Xtrain=np.zeros((300,1))
    for i in range(300):
        Xtrain[i]=xtrain[i]

    Ytrain=np.sin(Xtrain)


    rng = np.random.RandomState(23455)

    x=T.matrix('x')
    y=T.matrix('y')

    [hout, params_1] = cnn.fullyConnectedLayer(
        rng=rng,
        data_input=x,
        num_in=1,
        num_out=4)

    [y_pred_lin, params_2] = cnn.fullyConnectedLayer(
        rng=rng,
        data_input=T.tanh(hout),
        num_in=4,
        num_out=1)
    y_pred=T.tanh(y_pred_lin)

    cost=cnn.MSE(y,y_pred)

    params = params_1 + params_2

    updates = cnn.gradient_updates_Adam(cost,params,0.05)

    train = theano.function(
        inputs=[x,y],
        outputs=[cost],
        updates=updates)
    
    for i in range(2000):
        cost_i=train(Xtrain,Ytrain)
    assert cost_i[0] < 0.015

def test_gradient_updates_Adam():
    # Find minimum of a parabola
    x = T.matrix('x')
    w = theano.shared(100.0,borrow=True)
    h = T.dot(x,w)
    cost=T.mean(h**2)
    
    updates=cnn.gradient_updates_Adam(cost,[w],10)
    
    f=theano.function([x],cost,updates=updates)
    for i in range(100):
        cost_i=f(np.ones((1,1)))
    assert cost_i < 0.06
    
    x2 = T.matrix('x2')
    w2 = theano.shared(10.0,borrow=True)
    h2 = T.dot(x2,w2)
    cost2 = T.mean(T.sin(h2)**2+0.1*h2**2)

    updates2=cnn.gradient_updates_Adam(cost2,[w2],0.1)

    f2=theano.function([x2],cost2,updates=updates2)
    for i in range(200):
        cost_i2=f2(np.ones((1,1)))
    assert cost_i2 < 1
