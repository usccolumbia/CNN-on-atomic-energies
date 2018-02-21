import sys
import numpy as np
import theano
import theano.tensor as T
import load_data
import random as rd
import run_cnn
import hyppar
import datapar

print('******* Import complete *******')

rd.seed()

# Read input file
print("Reading input data...\n")
hyppar.setInput()

# Set up neural network structure
print("\n Setting up CNN structure...")
hyppar.setStructureParameters()

# Read raw data
print("Reading raw data...")
datapar.loadRawData()

# Handle dataset
print("Loading dataset...")
datapar.loadDataPoints()

# Define training, validation and test sets
print("Splitting dataset...")
datapar.splitDataset()

run_cnn.TrainCNN()
