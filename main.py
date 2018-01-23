import numpy as np
import load_data
#import plotting
import time
import random as rd
import run_cnn

rd.seed(23455)

Ndata=2400
Ntrain=2000
Nval=Ndata-Ntrain

alpha  = 0.001   # Adam Learning rate
Nepoch = 20      # Number of epochs
Nf     = [20,20] # Number of filters in each convlayer
mbs    = 50      # minibatch size
reg    = 0.0001  # Regularization parameter

###### Location of data ######
datapath = '/u/82/simulak1/unix/Desktop/kurssit/deep_learning/project/data'
filename = datapath+'/train.csv'
############################## 

#######  Get all available raw data ###### 
print("*** Loading raw data ***")
start=time.time()
spacegrp,Natoms,pc_al,pc_ga,pc_in,lv_alpha,lv_beta,lv_gamma,lvadeg,lvbdeg,lvgdeg,Ef,Eg = load_data.get_train_data(filename)
end=time.time()
print("Time taken to load data: "+str(end-start)+"\n")
##########################################

###### Get geometry data ###### 
print("*** Loading geometry data ***")
start=time.time()
xyz_Train,elements_Train,lattices_Train=load_data.get_geometry(Ntrain,datapath)
end=time.time()
print("Time taken to load geometry data: "+str(end-start)+"\n")
################################


###### Get all the relevant data for training ######
print('*** Loading datapoints ***')
Xtrain=np.zeros((Ntrain,6400))
Ytrain=np.zeros((Ntrain,))
for i in np.arange(Ntrain):
    X=np.loadtxt('CM/'+str(i+1)+'.gz')
    Xtrain[i,:]=X.flatten(0)
    Ytrain[i]=Ef[i]
Xval=np.zeros((Nval,6400))
Yval=np.zeros((Nval,))
for i in np.arange(Nval):
    X=np.loadtxt('CM/'+str(Ntrain+i+1)+'.gz')
    Xval[i,:]=X.flatten(0)
    Yval[i]=Ef[i]
timing=time.time()
print('Time taken to load data: '+str(timing-end))
####################################################

###### Turn data into theano shared variables ##################
valid_set_x, valid_set_y, valid_set = load_data.shared_dataset(
    Xval, Yval,
    sample_size=Nval)
train_set_x, train_set_y, train_set = load_data.shared_dataset(
    Xtrain, Ytrain,
    sample_size=Ntrain)
###############################################################


run_cnn.TrainCNN(train_set_x,train_set_y,valid_set_x,valid_set_y,alpha,Nepoch,Nf,mbs,reg)
