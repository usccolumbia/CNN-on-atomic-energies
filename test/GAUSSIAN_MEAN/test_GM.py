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

def gaussian(x,y):
    return np.exp(-(x**2+y**2)/10)

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

    x = np.zeros((len(np.arange(-0.5,0.5,0.002)),1)) # Datapoints
    y = np.zeros((len(np.arange(-0.5,0.6,0.1)),1))     # Features
    x[:,0]=np.arange(-0.5,0.5,0.002)
    y[:,0]=np.arange(-0.5,0.6,0.1)
    Nx=len(x)
    Ny=len(y)

    grid = np.zeros((Nx,Ny))

    for i in range(Nx):
        for j in range(Ny):
            grid[i,j] = gaussian(x[i],y[j])

    datapar.Ydata = np.zeros((3*Nx,1))
    datapar.Xdata = np.zeros((3*Nx,Ny,1))
    for i in range(Nx):
        X = grid[i,:]
        Y = np.mean(X)
        datapar.Ydata[i,0]      = Y
        datapar.Ydata[i+Nx,0]   = Y
        datapar.Ydata[i+2*Nx,0] = Y
        datapar.Xdata[i,:,0]      = X
        datapar.Xdata[i+Nx,:,0]   = X
        datapar.Xdata[i+2*Nx,:,0] = X

    datapar.splitDataset()
        
    if(hyppar.task=='classification'):
        train_classification.TrainCNN()
    else:
        train_regression.TrainCNN()

    assert(hyppar.final_valid_error<0.00000007)
