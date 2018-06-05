import numpy as np
from numpy import linalg as LA
import theano
import theano.tensor as T
import random as rd
import ../../load_data



def test_shared_dataset():
    # Below AB is not tested, it returns list of input arrays
    A=np.ones((2,2))
    B=np.ones((2,1))
    C=np.ones((2,))
    As,Bs,AB=load_data.shared_dataset(A,B,sample_size=2)
    As,Cs,AC=load_data.shared_dataset(A,B,sample_size=2)
    same=True
    for i in range(2):
        for j in range(2):
            if(np.abs(A[i,j]-As.get_value(borrow=True)[i,j])>0.01):
                same=False
    for i in range(2):
        if(np.abs(B[i,0]-Bs.get_value(borrow=True)[i,0])>0.01):
            same=False
    for i in range(2):
        if(np.abs(C[i]-Cs.get_value(borrow=True)[i])>0.01):
            same=False
           
    
    assert same
            









