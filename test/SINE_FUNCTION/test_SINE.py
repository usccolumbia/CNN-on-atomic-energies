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
    datapar.loadDataPoints()
    
    # Define training, validation and test sets
    print("Splitting dataset...")
    datapar.splitDataset()
    
    if(hyppar.task=='classification'):
        train_classification.TrainCNN()
    else:
        train_regression.TrainCNN()

    assert(hyppar.final_valid_error<0.00001)
