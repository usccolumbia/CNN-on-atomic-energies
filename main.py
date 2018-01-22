import numpy as np
import load_data
import plotting
import time
import matplotlib.pyplot as plt

# Location of data
datapath = '/u/82/simulak1/unix/Desktop/kurssit/deep_learning/project/data'
filename = datapath+'/train.csv'

Ntrain=10

# Get all available raw data
print("*** Loading raw data ***")
start=time.time()
spacegrp,Natoms,pc_al,pc_ga,pc_in,lv_alpha,lv_beta,lv_gamma,lvadeg,lvbdeg,lvgdeg,Ef,Eg = load_data.get_train_data(filename)
end=time.time()
print("Time taken to load data: "+str(end-start)+"\n")

print("*** Loading geometry data ***")
start=time.time()
xyz_Train,elements_Train,lattices_Train=load_data.get_geometry(Ntrain,datapath)
end=time.time()
print("Time taken to load geometry data: "+str(end-start)+"\n")


#plotting.EnergyHist(Natoms,'blue','Histogram of atom numbers','Number of atoms in unit cell')

# Make Coulomb matrices
print("*** Making Coulomb matrices ***")
start=time.time()
CM_Train = np.zeros((Ntrain,80,80))
for i in np.arange(Ntrain):
    CM_Train[i,:,:]=load_data.make_correlation_matrix(xyz_Train[i],lattices_Train[i],elements_Train[i])
end=time.time()
print("Time taken to make Coulomb matrices: "+str(end-start)+"\n")##

for i in np.arange(Ntrain):
    np.savetxt('CM/'+str(i)+'.gz',CM_Train[i,:,:])

