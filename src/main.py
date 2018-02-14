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
print("Done.")

# Read raw data
print("Reading raw data...")
datapar.loadRawData()
print("Done.")


# Handle dataset
print("Loading dataset...")
datapar.loadDataPoints()
print("Done.")

######### Split data into training and validation sets #########
Xtrain=np.zeros((hyppar.Ntrain,6400))
Ytrain=np.zeros((hyppar.Ntrain,1))
Xval=np.zeros((hyppar.Nval,6400))
Yval=np.zeros((hyppar.Nval,1))
Xtest=np.zeros((600,6400))
for i in range(hyppar.Nval):
    Xval[i,:]=datapar.Xdata[i,:,:].flatten(0)
    Yval[i,0]=datapar.Ydata[i,0]
for i in range(hyppar.Ntrain):
    ind=hyppar.Nval+i
    Xtrain[i,:]=datapar.Xdata[ind,:,:].flatten(0)
    Ytrain[i,0]=datapar.Ydata[ind,0]
for i in range(600):
    X=np.loadtxt(hyppar.datapath+'CM/test/'+str(i+1))
    Xtest[i,:]=X.flatten(0)
################################################################


###### Turn data into theano shared variables ##################
valid_set_x, valid_set_y, valid_set = load_data.shared_dataset(
    Xval, Yval,
    sample_size=hyppar.Nval)
train_set_x, train_set_y, train_set = load_data.shared_dataset(
    Xtrain, Ytrain,
    sample_size=hyppar.Ntrain)
test_set_x = load_data.shared_testset(Xtest) 
###############################################################

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

