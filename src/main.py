import sys
import numpy as np
import theano
import theano.tensor as T
import load_data
#import plotting
import time
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


print(hyppar.Nchannel[9])
# Read raw data
print("Reading raw data...")
datapar.loadRawData()

# Handle dataset
print("Loading dataset...")
datapar.loadDataPoints()

# Define training, validation and test sets
print("Splitting dataset...")
datapar.splitDataset()

dir='output'
np.savetxt(dir+'/E_target.txt',Yval)
Et,Ev,w0_arr1,w0_arr2,w0_arr3, w1_arr1,w1_arr2,w1_arr3,w2_arr1,w2_arr2,wf1_arr1,wf1_arr2,wf1_arr3,E_test= run_cnn.TrainCNN(train_set_x,train_set_y,valid_set_x,valid_set_y,test_set_x,hyppar.learning_rate,hyppar.Nepoch,hyppar.Nchannel,hyppar.mbs,hyppar.reg)

np.savetxt(dir+'/et.txt',Et)
np.savetxt(dir+'/ev.txt',Ev)

np.savetxt(dir+'/w0_sample1.txt',w0_arr1)
np.savetxt(dir+'/w0_sample2.txt',w0_arr2)
np.savetxt(dir+'/w0_sample3.txt',w0_arr3)

np.savetxt(dir+'/w1_sample1.txt',w1_arr1)
np.savetxt(dir+'/w1_sample2.txt',w1_arr2)
np.savetxt(dir+'/w1_sample3.txt',w1_arr3)

np.savetxt(dir+'/w2_sample1.txt',w2_arr1)
np.savetxt(dir+'/w2_sample2.txt',w2_arr2)

np.savetxt(dir+'/wfc1_sample1.txt',wf1_arr1)
np.savetxt(dir+'/wfc1_sample2.txt',wf1_arr2)
np.savetxt(dir+'/wfc1_sample3.txt',wf1_arr3)

np.savetxt(dir+'/E_test.txt',E_test)

