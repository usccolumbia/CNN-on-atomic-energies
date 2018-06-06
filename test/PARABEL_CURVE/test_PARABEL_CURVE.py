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
import statistics
hyppar.current_dir=orig_path

print('******* Import complete *******')
def test_SINE():
    rd.seed()

    # Read input file
    print("Reading input data...\n")
    hyppar.setInput(orig_path+"/input")
    hyppar.datapath=hyppar.current_dir+"/data" # TEST dir modification necessities
    
    # Set up neural network structure
    print("\n Setting up CNN structure...")
    hyppar.setStructureParameters()
    
    x=np.zeros((len(np.arange(-28,28,0.5)),1))
    N=len(x)
    x[:,0]=np.arange(-28,28,0.5)
    datapar.Xdata=np.zeros((3*N,1,1))
    for i in range(N):
        datapar.Xdata[i,0,0]     =x[i]
        datapar.Xdata[i+N,0,0]   =x[i]
        datapar.Xdata[i+2*N,0,0] =x[i]
    #y=np.multiply(np.sin(x3),x3**2)

    datapar.Ydata=np.multiply(np.sin(datapar.Xdata),datapar.Xdata**2)

    print(datapar.Xdata.shape)
    # Define training, validation and test sets
    print("Splitting dataset...")
    datapar.splitDataset()
    
    if(hyppar.task=='classification'):
        train_classification.TrainCNN()
    else:
        train_regression.TrainCNN()

    assert(hyppar.final_valid_error<400)
    statistics.w=[]
    statistics.b=[]
    statistics.conv_out=[]
    statistics.fc_out=[]
