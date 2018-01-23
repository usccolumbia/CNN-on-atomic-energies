import numpy as np
import load_data
import plotting
import time
import matplotlib.pyplot as plt

###### Location of data ######
datapath = '/u/82/simulak1/unix/Desktop/kurssit/deep_learning/project/data'
filename = datapath+'/train.csv'
############################## 
Ntrain=100

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

#plotting.EnergyHist(Natoms,'blue','Histogram of atom numbers','Number of atoms in unit cell')

###### Make Coulomb matrices ###### 
#print("*** Making Coulomb matrices ***")
#start=time.time()
#CM_Train = np.zeros((Ntrain,80,80))
#for i in np.arange(Ntrain):
#    CM_Train[i,:,:]=load_data.make_correlation_matrix(xyz_Train[i],lattices_Train[i],elements_Train[i])
#end=time.time()
#print("Time taken to make Coulomb matrices: "+str(end-start)+"\n")##
####################################

#for i in np.arange(Ntrain):
#    np.savetxt('CM/'+str(i)+'.gz',CM_Train[i,:,:])

testind=90
a=np.loadtxt("CM/"+str(testind)+".gz")
b=np.loadtxt("CM/"+str(testind)+"_new.gz")
count=0
pp=0

for i in np.arange(int(Natoms[testind])):
    for j in np.arange(i,int(Natoms[testind])):
        pp=pp+1
        if(np.abs(a[i,j]-b[i,j])>1):
            print("*** MOTHERFUCKER!!! ***")
            print('a: '+str(a[i,j])+', b: '+str(b[i,j]))
            count=count+1
print(pp)
print(count)
