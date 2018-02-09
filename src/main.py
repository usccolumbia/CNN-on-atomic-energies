import sys
print(sys.path)#.append("/homeappl/home/krsimula/appl_taito/src/ZOFCNN/src/")
import numpy as np
import theano
import theano.tensor as T
import load_data
#import plotting
import time
import random as rd
import run_cnn

print('******* Import complete *******')

rd.seed()

Ndata=2
Naug=2

Ntrain=2
Nval=2

alpha  = 0.0005   # Adam Learning rate
Nepoch = 10     # Number of epochs
Nf     = [10,10,10] # Number of filters in each convlayer
mbs    = 1      # minibatch size
reg    = 0.01  # Regularization parameter   %  

###### Location of data ######
datapath = '/wrk/krsimula/DONOTREMOVE/NEURAL_NETWORKS/CNN-on-atomic-energies/'
filename = datapath+'data/train.csv'
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
xyz_Train,elements_Train,lattices_Train=load_data.get_geometry(Ndata,datapath+'data')
end=time.time()
print("Time taken to load geometry data: "+str(end-start)+"\n")
################################


###### Download and augment Coulomb matrices ######
print('*** Loading and augmenting datapoints ***')
Xdata=np.zeros((Ndata+Naug,80,80))
Ydata=np.zeros((Ndata+Naug,1))
step=0
for i in np.arange(Ndata):
    X=np.loadtxt(datapath+'CM/train/'+str(i+1)+'.gz')
    XX = load_data.RandomSort(X,0.15)
    Xdata[step,:,:]=XX
    Ydata[step,0]=100*Ef[i]
    step=step+1
    
for i in np.arange(Naug):
    j = rd.randrange(0,Ndata)
    X=np.loadtxt(datapath+'CM/train/'+str(j+1)+'.gz')
    XX = load_data.RandomSort(X,0.15)
    Xdata[step,:,:]=XX
    Ydata[step,0]=100*Ef[j]
    step=step+1

timing=time.time()
print('Time taken to load data: '+str(timing-end))
####################################################

######### Split data into training and validation sets #########
Xtrain=np.zeros((Ntrain,6400))
Ytrain=np.zeros((Ntrain,1))
Xval=np.zeros((Nval,6400))
Yval=np.zeros((Nval,1))
Xtest=np.zeros((600,6400))
for i in range(Nval):
    Xval[i,:]=Xdata[i,:,:].flatten(0)
    Yval[i,0]=Ydata[i,0]
for i in range(Ntrain):
    ind=Nval+i
    Xtrain[i,:]=Xdata[ind,:,:].flatten(0)
    Ytrain[i,0]=Ydata[ind,0]
for i in range(600):
    X=np.loadtxt(datapath+'CM/test/'+str(i+1))
    Xtest[i,:]=X.flatten(0)
################################################################


###### Turn data into theano shared variables ##################
valid_set_x, valid_set_y, valid_set = load_data.shared_dataset(
    Xval, Yval,
    sample_size=Nval)
train_set_x, train_set_y, train_set = load_data.shared_dataset(
    Xtrain, Ytrain,
    sample_size=Ntrain)
test_set_x = load_data.shared_testset(Xtest) 
###############################################################

dir='output'
np.savetxt(dir+'/E_target.txt',Yval)
Et,Ev,w0_arr1,w0_arr2,w0_arr3, w1_arr1,w1_arr2,w1_arr3,w2_arr1,w2_arr2,wf1_arr1,wf1_arr2,wf1_arr3,E_test= run_cnn.TrainCNN(train_set_x,train_set_y,valid_set_x,valid_set_y,test_set_x,alpha,Nepoch,Nf,mbs,reg)

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

