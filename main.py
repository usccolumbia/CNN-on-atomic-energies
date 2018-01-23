import numpy as np
import load_data
import plotting
import time
import matplotlib.pyplot as plt

###### Location of data ######
datapath = '/u/82/simulak1/unix/Desktop/kurssit/deep_learning/project/data'
filename = datapath+'/train.csv'
############################## 

#######  Get all available raw data ###### 
Ntrain=2400

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
Xtrain=np.zeros((Ntrain,80,80))
Ytrain=np.zeros((Ntrain,))
for i in np.arange(Ntrain):
    Xtrain[i,:,:]=np.loadtxt('CM/'+str(i+1)+'.gz')
    Ytrain[i]=Ef[i]
timing=time.time()
print('Time taken to load data: '+str(timing-end))
####################################################

