import numpy as np
import hyppar
import load_data
import random as rd

# Training set
global Xtrain
# Training set labels
global Ytrain
# Validation set
global Xval
# Validation set labels
global Yval
# Test set features
global Xtest
# Test set labels
global Ytest

global Xdata
global Ydata

def splitDataset():
    '''
    Splits dataset into training and validation sets based on earlier
    defined variables Ntrain and Nval. Currently test set is included
    separately.
    '''
    # New variables:
    # Training set
    global Xtrain
    # Training set labels
    global Ytrain
    # Validation set
    global Xval
    # Validation set labels
    global Yval
    # Test set features
    global Xtest
    # Test set labels
    global Ytest
    
    # Use: from module hyppar
    Ntrain   = hyppar.Ntrain
    Nval     = hyppar.Nval
    Ntest    = hyppar.Ntest
    datapath = hyppar.datapath
    xdim     = hyppar.in_x
    ydim     = hyppar.in_y
    
    # Use: local module
    global Xdata
    global Ydata

    Xtrain=np.zeros((Ntrain,xdim*ydim))
    Ytrain=np.zeros((Ntrain,1))
    Xval=np.zeros((Nval,xdim*ydim))
    Yval=np.zeros((Nval,1))
    Xtest=np.zeros((Ntest,xdim*ydim))
    Ytest=np.zeros((Ntest,1))
    
    if (hyppar.target_type=='int'):
        for i in range(Nval):
            Xval[i,:]=Xdata[i,:,:].flatten(0)
            Yval[i,0]=int(Ydata[i,0])
        for i in range(Ntrain):
            ind=Nval+i
            Xtrain[i,:]=Xdata[ind,:,:].flatten(0)
            Ytrain[i,0]=int(Ydata[ind,0])
        for i in range(Ntest):
            ind=Nval+Ntrain+i
            Xtest[i,:]=Xdata[ind,:,:].flatten(0)
            Ytest[i,0]=int(Ydata[ind,0])
    else:
        for i in range(Nval):
            Xval[i,:]=Xdata[i,:,:].flatten(0)
            Yval[i,0]=float(Ydata[i,0])
        for i in range(Ntrain):
            ind=Nval+i
            Xtrain[i,:]=Xdata[ind,:,:].flatten(0)
            Ytrain[i,0]=float(Ydata[ind,0])
        for i in range(Ntest):
            ind=Nval+Ntrain+i
            Xtest[i,:]=Xdata[ind,:,:].flatten(0)
            Ytest[i,0]=float(Ydata[ind,0])
                                                                                            
        
    # ONEHOT_ENCODING: (Not currently used)
    if(hyppar.task=='classification'):
        Ytrain=load_data.onehot(Ytrain[:,0],hyppar.Nclass)
        Yval=load_data.onehot(Yval[:,0],hyppar.Nclass)
        Ytest=load_data.onehot(Ytest[:,0],hyppar.Nclass)
        
def loadDataPoints():
    '''
    Download feature matrices and target vector
    '''
    # Target value vector
    global Ydata
    global Xdata

    # Use: hyppar-module
    Ndata    = hyppar.Ndata
    datapath = hyppar.datapath
    xdim     = hyppar.in_x
    ydim     = hyppar.in_y
    zdim     = hyppar.in_z
    
    step=0
    
    Xdata=np.zeros((Ndata,xdim,ydim))
    for i in np.arange(Ndata):
        X=np.loadtxt(datapath+'/Xdata/'+str(i))
        if(xdim>1 and ydim==1):
            XX=np.zeros((xdim,1))
            XX[:,0]=X
            X=XX
        Xdata[step,:,:]=X
        step=step+1
        

    Ydata=np.zeros((Ndata,1))
    Ydata_vector=np.loadtxt(datapath+'/Ydata')
    for i in range(Ndata):
        Ydata[i,0]=Ydata_vector[i]

    

