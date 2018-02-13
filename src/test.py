import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import cnn


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


visualize_sinetraining(300,4,2000,0.05)
