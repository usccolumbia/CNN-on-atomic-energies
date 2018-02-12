import theano
import theano.tensor as T
import numpy as np
import cnn

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
