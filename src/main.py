import sys
import numpy as np
import theano
import theano.tensor as T
import load_data
import random as rd
import train_regression
import train_classification
import hyppar
import datapar

print('******* Import complete *******')

rd.seed()

hyppar.current_dir=sys.path[0]

# Read input file
print("Reading input data...\n")
hyppar.setInput()

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


