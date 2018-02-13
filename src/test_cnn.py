import theano
import theano.tensor as T
import numpy as np
import cnn

def test_fullyConnectedLayer(Npoints,Nnodes,Nsteps,learning_rate):
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
    assert cost_i < 0.015

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
