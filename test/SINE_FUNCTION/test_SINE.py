import sys
orig_path=sys.path[0]
split_path = sys.path[0].split('/')
newpath=""
for i in range(len(split_path)-2):
    newpath=newpath+'/'+split_path[i]
newpath=newpath+'/src'
sys.path[0] = newpath
import numpy as np
import theano
import theano.tensor as T
import load_data
import random as rd
import train_regression
import train_classification
import hyppar
import datapar
hyppar.current_dir=orig_path

print('******* Import complete *******')
def test_SINE():
    rd.seed()

    # Read input file
    print("Reading input data...\n")
    hyppar.setInput(orig_path+"/input")
    hyppar.datapath=hyppar.current_dir+"/data"
    
    # Set up neural network structure
    print("\n Setting up CNN structure...")
    hyppar.setStructureParameters()

    # Handle dataset
    print("Loading dataset...")
    Ntrain   = hyppar.Ntrain
    Nval     = hyppar.Nval
    Ntest    = hyppar.Ntest
    datapath = hyppar.datapath
    xdim     = hyppar.in_x
    ydim     = hyppar.in_y

    datapar.Xtrain=np.zeros((Ntrain,xdim*ydim))
    datapar.Ytrain=np.zeros((Ntrain,1))
    datapar.Xval=np.zeros((Nval,xdim*ydim))
    datapar.Yval=np.zeros((Nval,1))
    datapar.Xtest=np.zeros((Ntest,xdim*ydim))
    datapar.Ytest=np.zeros((Ntest,1))

    for i in range(Ntrain):
        x=i*2*3.14/Ntrain
        y=np.sin(x)
        datapar.Xtrain[i,0]=x
        datapar.Ytrain[i,0]=y
        datapar.Xval[i,0]=x
        datapar.Yval[i,0]=y
        datapar.Xtest[i,0]=x
        datapar.Ytest[i,0]=y
                                
    
    if(hyppar.task=='classification'):
        train_classification.TrainCNN()
    else:
        train_regression.TrainCNN()

    assert(hyppar.final_valid_error<0.00001)
