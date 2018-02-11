import numpy as np
from numpy import linalg as LA
import theano
import theano.tensor as T
import random as rd
import load_data

def test_RandomSort():

    ##### Test that array is permuted correctly ########
    X=np.array(((1,2,3),(2,0,0),(3,0,0)))
    XX=load_data.RandomSort(X,0.000001)

    Xf=np.array(((0,0,2),(0,0,3),(2,3,1)))

    same=True
    for i in range(3):
        for j in range(3):
            if(Xf[i,j]-XX[i,j]>0.01):
                same=False
                
    assert same

    X=np.array(((1,0,0),(0,5,0),(0,0,6)))
    XX=XX=load_data.RandomSort(X,0.000001)
    same=True
    for i in range(3):
        for j in range(3):
            if(X[i,j]-XX[i,j]>0.01):
                same=False

    assert same

    ########## Test that my method produces deviations ############
    X = np.random.normal(0,100,(50,50))
    changed=False
    for k in range(200):
        XX=load_data.RandomSort(X,10000)
        for i in range(3):
            for j in range(3):
                if(X[i,j]-XX[i,j]>0.01):
                    changed=True
    assert changed








