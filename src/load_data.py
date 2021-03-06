import numpy as np
from numpy import linalg as LA
import theano
import theano.tensor as T
import random as rd

def RandomSort(X,sigma):
    np.random.seed()
    
    xdim=X.shape[0]
    r=LA.norm(X,axis=1)
    r=r+np.random.normal(0.0,sigma,xdim)
    i=np.argsort(r)
    X1=X[:,i]
    return X1[i]

def shared_dataset(data_x, data_y, sample_size=2400, borrow=True):
    rd.seed(23455)
    indices = 0
    if (sample_size < 0):
        print('Sample size too small!')
        return
    try:
        indices = rd.sample(range(0, data_y.shape[0]), sample_size)
    except ValueError:
        print('Sample size exceeds data size.')
    data_x = data_x[indices, :]
    data_y = data_y[indices]
    
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, shared_y, (data_x, data_y)

def shared_testset(data_x, sample_size=600, borrow=True):
    rd.seed(23455)
    indices = 0
    if (sample_size < 0):
        print('Sample size too small!')
        return
    try:
        indices = rd.sample(range(0, sample_size), sample_size)
    except ValueError:
        print('Sample size exceeds data size.')
    x = np.zeros((sample_size,6400))
    x = data_x[indices, :]
        
    shared_x = theano.shared(np.asarray(x,
                                        dtype=theano.config.floatX),
                             borrow=borrow)

    return shared_x


def make_correlation_matrix_fast(xyz,lat,elm,rndnoise=False):
    '''
    Constructs a Coulomb matrix for a provided system.

    Input: 
    * xyz   : <list of np.array((3,))>             , coordinates of each atom
    * lat   : <list of np.array((3,),dtype=float)> , lattice vectors
    * elm   : <list of strings>                    , elements of each atom

    Output:
    * CM    : np.array((80,80))         , Coulomb matrix

    NOTE: CM is symmetric, hence compute only upper triangle!
    '''

    # Number of atoms
    N = len(xyz)

    # Initialize Coulomb matrix, this takes care of padding also
    CM = np.zeros((80,80))

    # lattive vectors
    A=lat[0]
    B=lat[1]
    C=lat[2]
    Anorm=LA.norm(A)
    Bnorm=LA.norm(B)
    Cnorm=LA.norm(C)
    
    
    # Construct coordinates in lattice vector basis
    xyz_rel=np.zeros((N,3))
    for i in np.arange(N):
        xyz_rel[i,0]=np.dot(xyz[i],A)/Anorm
        xyz_rel[i,1]=np.dot(xyz[i],B)/Bnorm
        xyz_rel[i,2]=np.dot(xyz[i],C)/Cnorm
    
    # Atomic charges of our elements
    Z=np.zeros((N,))
    for i in np.arange(N):
        if(elm[i]=="Ga"):
            Z[i]=31
        elif(elm[i]=="Al"):
            Z[i]=13
        elif(elm[i]=="In"):
            Z[i]=49
        elif(elm[i]=="O"):
            Z[i]=8
        else:
            print("######### Error: Atom type not found #########")
            

    for i in np.arange(N):
        ri = xyz[i]
        ri_rel=xyz_rel[i,:]
        for j in np.arange(i,N):
            if(i==j):
                CM[i,i]=0.5*(Z[i]**2.4)
            else:
                rj = xyz[j]
                rj_rel=xyz_rel[j,:]

                xx=1; yy=1; zz=1 # Assumption at first: i:th atom closer to origin
                
                dd=ri-rj
                
                if(ri_rel[0]>rj_rel[0]):
                    xx=-1
                if(ri_rel[1]>rj_rel[1]):
                    yy=-1
                if(ri_rel[2]>rj_rel[2]):
                    zz=-1

                AA=xx*A
                BB=yy*B
                CC=zz*C
                    
                d1=LA.norm(ri-rj)

                d2=LA.norm(AA+dd)
                d3=LA.norm(BB+dd)
                d4=LA.norm(CC+dd)

                d5=LA.norm(AA+BB+dd)
                d6=LA.norm(AA+CC+dd)
                d7=LA.norm(BB+CC+dd)


                d8=LA.norm(AA+BB+CC+dd)

                d=np.amin(np.array((d1,d2,d3,d4,d5,d6,d7,d8)))
                    
                CM[i,j]=Z[i]*Z[j]/d
                CM[j,i]=CM[i,j]

    if(rndnoise):
        return CM
    else:
        return CM
    
def get_xyz_data(filename):
    '''
    A function for reading in geometric information on different systems.

    Input: filename of the file with atom coordinates and element numbers.

    Output: 
    * pos_data: type=tuple(float([x,y,z]),str(a))  , Includes coordinates and atom element. 
    * lat_data: type=np.array((3,3),dtype='float') , includes lattice vectors as rows
    '''

    pos_data = []
    lat_data = []
    
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                pos_data.append([np.array(x[1:4], dtype=np.float),x[4]])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))

    return pos_data, np.array(lat_data)
        
def get_train_data(filename):
    '''
    Extract all the provided data for systems as lists of ints or floats.
    Important outputs:
    * Eg: list of all the band gap energies of the systems
    * Ef: list of all the formation energies of the system
    '''

    spacegrp = []
    Natoms   = []
    pc_al    = []
    pc_ga    = []
    pc_in    = []
    lv_alpha = []
    lv_beta  = []
    lv_gamma = []
    lvadeg   = []
    lvbdeg   = []
    lvgdeg   = []
    Ef       = []
    Eg       = []

    with open(filename) as f:
        
        for line in f.readlines():
            x = line.split(",")
            if(not(x[0]=='id')):
                spacegrp.append(int(x[1]))
                Natoms.append(float(x[2]))
                pc_al.append(float(x[3]))
                pc_ga.append(float(x[4]))
                pc_in.append(float(x[5]))
                lv_alpha.append(float(x[6]))
                lv_beta.append(float(x[7]))
                lv_gamma.append(float(x[8]))
                lvadeg.append(float(x[9]))
                lvbdeg.append(float(x[10]))
                lvgdeg.append(float(x[11]))
                Ef.append(float(x[12]))
                Eg.append(float(x[13]))
                
    return spacegrp,Natoms,pc_al,pc_ga,pc_in,lv_alpha,lv_beta,lv_gamma,lvadeg,lvbdeg,lvgdeg,Ef,Eg
                                    
    
def get_geometry(Ntrain,datapath):
    '''
    Returns geometry information of the training data.
    Inputs:
    * Ntrain    : <int>   , number of training data points (Max 2400)
    * datapath  : <float> , path to the directory including the geometry info
    Outputs:
    * xyz_train : <list of list of np.array((3,0))> , coordinates of all of the atoms  
    * elements  : <list of list of str values>      , elements of all the atoms
    * lattices  : <list of list of np.array((3,3))> , all lattice coordinates 
    '''
    xyz_train = []
    elements  = []
    lattices  = []

    for i in np.arange(Ntrain):
        index=i+1
        pos,lat=get_xyz_data(datapath+'/train/'+str(index)+'/geometry.xyz')
        xyz=[(tup[0]) for tup in pos]
        xyz_train.append(xyz)
        elm=[(tup[1]) for tup in pos]
        elements.append(elm)
        lattices.append(lat)
        
    return xyz_train,elements,lattices
                                                                            
