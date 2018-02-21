import numpy as np
import hyppar
import load_data
import random as rd

def splitDataset():
    '''
    Splits dataset into training and validation sets based on earlier
    defined variables Ntrain and Nval. Currently test set is included
    separately.
    '''
    # New variables:
    # Training set
    global Xtrain
    # Training set labels
    global Ytrain
    # Validation set
    global Xval
    # Validation set labels
    global Yval
    # Test set: Only features
    global Xtest

    # Use: from module hyppar
    Ntrain=hyppar.Ntrain
    Nval=hyppar.Nval
    datapath=hyppar.datapath
    
    # Use: local module
    global Xdata
    global Ydata

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



def loadDataPoints():
    '''
    Loads feature vectors as lists of Coulomb matrices and
    labels as a N x 1 matrix of formation (or band gap) energies.
    '''
    rd.seed()
    # Dataset features
    global Xdata
    # Dataset labels
    global Ydata

    # Use: local module variables
    global Ef

    # Use: hyppar-module
    Ndata    = hyppar.Ndata
    Naug     = hyppar.Naug
    datapath = hyppar.datapath
    


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


def loadRawData():
    '''
    Download global variables of systems.
    Each variable is a list of values for each system.
    Coulomb matrices follow the same order in indexing.
    '''
    # Spacegroup of the systems
    global spacegrp
    # Number of atoms
    global Natoms
    # Percentage of Al
    global pc_al
    # % Ga
    global pc_ga
    # % In
    global pc_in
    global lv_alpha
    global lv_beta
    global lv_gamma
    global lvadeg
    global lvbdeg
    global lvgdeg
    # Formation energies LABELS
    global Ef
    # Band gap energies LABELS
    global Eg
    # Training set atom coordinates
    global xyz_Train
    # Training set atom elements (str)
    global elements_Train
    # Training set lattice vectors
    global lattices_Train

    filename = hyppar.datapath+'data/train.csv'
    spacegrp,Natoms,pc_al,pc_ga,pc_in,lv_alpha,lv_beta,lv_gamma,lvadeg,lvbdeg,lvgdeg,Ef,Eg = load_data.get_train_data(filename)
    xyz_Train,elements_Train,lattices_Train=load_data.get_geometry(hyppar.Ndata,hyppar.datapath+'data')

