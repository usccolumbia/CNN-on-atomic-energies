import numpy as np
from numpy import linalg as LA
import theano
import theano.tensor as T
import random as rd
import load_data

def test_RandomSort():
    X=np.array(((1,3,2),(3,0,0),(2,0,1)))
#Xnorm = np.array((np.sqrt(14),3,np.sqrt(5)))
#i=np.array((0,1,2))
    XX=load_data.RandomSort(X,00000000000001)

    same=True
    for i in range(3):
        for j in range(3):
            if(X[i,j]-XX[i,j]>0.01):
                same=False
                
    assert False



#XX=load_data.RandomSort(X,0.0000000001)


