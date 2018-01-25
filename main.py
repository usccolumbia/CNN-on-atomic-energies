import numpy as np
import theano
import theano.tensor as T
import load_data
#import plotting
import time
import random as rd
import run_cnn

rd.seed(23455)

Ndata=100
Ntrain=60
Nval=Ndata-Ntrain

alpha  = 0.0001   # Adam Learning rate
Nepoch = 3      # Number of epochs
Nf     = [10,10,20] # Number of filters in each convlayer
mbs    = 5      # minibatch size
reg    = 0.001  # Regularization parameter

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
Ytrain=np.zeros((Ntrain,1))
for i in np.arange(Ntrain):
    X=np.loadtxt('CM/'+str(i+1)+'.gz')
    Xtrain[i,:]=X.flatten(0)
    Ytrain[i,0]=Ef[i]
Xval=np.zeros((Nval,6400))
Yval=np.zeros((Nval,1))
for i in np.arange(Nval):
    X=np.loadtxt('CM/'+str(Ntrain+i+1)+'.gz')
    Xval[i,:]=X.flatten(0)
    Yval[i,0]=Ef[i]
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


Et,Ev,w0_1,w0_2,w0_3, w1_arr1,w1_arr2,w1_arr3,w2_arr1,w2_arr2,w2_arr3,wf_arr1,wf_arr2,wf_arr3= run_cnn.TrainCNN(train_set_x,train_set_y,valid_set_x,valid_set_y,alpha,Nepoch,Nf,mbs,reg)

print(w0_1.shape)
