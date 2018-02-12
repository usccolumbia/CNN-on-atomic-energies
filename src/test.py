import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import cnn

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
        num_out=3)

[y_pred, params_2] = cnn.fullyConnectedLayer(
    rng=rng,
    data_input=hout,
    num_in=3,
    num_out=1)

cost=cnn.MSE(y,y_pred)

params = params_1 + params_2

updates = cnn.gradient_updates_Adam(cost,params,0.05)

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

for i in range(10000):
    pred,cost_i,hout=train(Xtrain,Ytrain)
    if(i%50==0):
        ax1.clear()
        ax1.plot(xtrain,ytrain,'b--')
        ax1.plot(Xtrain,pred,'r-')
        #ax2.plot(i,cost,'r+')
        ax3.clear()
        ax3.plot(Xtrain,hout)
        fig.canvas.draw()
        plt.pause(0.05)
