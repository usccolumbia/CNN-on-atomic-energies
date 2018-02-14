import numpy as np
from numpy import linalg as LA
import theano
import theano.tensor as T
import random as rd
import load_data



def test_shared_testset():
    A=np.ones((2,2))
    As=load_data.shared_testset(A,sample_size=2)
    same=True
    for i in range(2):
        for j in range(2):
            if(np.abs(A[i,j]-As.get_value(borrow=True)[i,j])>0.01):
                same=False
    
    assert same

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
            

def test_make_correlation_matrix_fast():

    ### Reference matrix ####
    CM_manual=np.zeros((2,2))
    
    Z_Ga = 31
    Z_Al = 13
    d=np.sqrt(2)

    CM_manual[0,0] = 0.5*(Z_Ga**2.4)
    CM_manual[1,1] = 0.5*(Z_Al**2.4)
    CM_manual[0,1] = Z_Ga*Z_Al/d
    CM_manual[1,0] = Z_Ga*Z_Al/d
    ###########################
    
    # 1)
    xyz=[]
    elm=[]
    lat=[]
    
    r1=np.array((1.5,1.5,0))
    r2=np.array((2.5,2.5,0))
    
    l1=np.array((4,0,0))
    l2=np.array((0,4,0))
    l3=np.array((0,0,100))
    
    elm1="Ga"
    elm2="Al"

    xyz.append(r1)
    xyz.append(r2)

    lat.append(l1)
    lat.append(l2)
    lat.append(l3)

    elm.append(elm1)
    elm.append(elm2)

    CM=load_data.make_correlation_matrix_fast(xyz,lat,elm)
    same=True
    for i in range(2):
        for j in range(2):
            if (np.abs(CM_manual[i,j]-CM[i,j])>0.001):
                print(CM_manual)
                print(CM)
                same=False
                
    assert same
    # 2)                                                                                                          
    xyz=[]
    elm=[]
    lat=[]

    r1=np.array((1.5,2.5,0))
    r2=np.array((2.5,1.5,0))

    l1=np.array((4,0,0))
    l2=np.array((0,4,0))
    l3=np.array((0,0,100))

    elm1="Ga"
    elm2="Al"

    xyz.append(r1)
    xyz.append(r2)

    lat.append(l1)
    lat.append(l2)
    lat.append(l3)

    elm.append(elm1)
    elm.append(elm2)

    CM=load_data.make_correlation_matrix_fast(xyz,lat,elm)
    same=True
    for i in range(2):
        for j in range(2):
            if (np.abs(CM_manual[i,j]-CM[i,j])>0.001):
                same=False

    assert same
    # 3)                                                                                                          
    xyz=[]
    elm=[]
    lat=[]

    r1=np.array((0,1.5,1.5))
    r2=np.array((0,2.5,2.5))

    l1=np.array((100,0,0))
    l2=np.array((0,4,0))
    l3=np.array((0,0,4))

    elm1="Ga"
    elm2="Al"

    xyz.append(r1)
    xyz.append(r2)

    lat.append(l1)
    lat.append(l2)
    lat.append(l3)

    elm.append(elm1)
    elm.append(elm2)

    CM=load_data.make_correlation_matrix_fast(xyz,lat,elm)
    same=True
    for i in range(2):
        for j in range(2):
            if (np.abs(CM_manual[i,j]-CM[i,j])>0.001):
                same=False
    assert same
                
    # 4)                                                                                                          
    xyz=[]
    elm=[]
    lat=[]

    r1=np.array((0,2.5,1.5))
    r2=np.array((0,1.5,2.5))

    l1=np.array((100,0,0))
    l2=np.array((0,4,0))
    l3=np.array((0,0,4))

    elm1="Ga"
    elm2="Al"

    xyz.append(r1)
    xyz.append(r2)

    lat.append(l1)
    lat.append(l2)
    lat.append(l3)

    elm.append(elm1)
    elm.append(elm2)

    CM=load_data.make_correlation_matrix_fast(xyz,lat,elm)
    same=True
    for i in range(2):
        for j in range(2):
            if (np.abs(CM_manual[i,j]-CM[i,j])>0.001):
                same=False
    assert same

    # 5)                                                                                                          
    xyz=[]
    elm=[]
    lat=[]

    r1=np.array((0.5,0.5,0))
    r2=np.array((3.5,3.5,0))

    l1=np.array((4,0,0))
    l2=np.array((0,4,0))
    l3=np.array((0,0,100))

    elm1="Ga"
    elm2="Al"

    xyz.append(r1)
    xyz.append(r2)

    lat.append(l1)
    lat.append(l2)
    lat.append(l3)

    elm.append(elm1)
    elm.append(elm2)

    CM=load_data.make_correlation_matrix_fast(xyz,lat,elm)
    same=True
    for i in range(2):
        for j in range(2):
            if (np.abs(CM_manual[i,j]-CM[i,j])>0.001):
                same=False
    assert same

    # 6)                                                                                                          
    xyz=[]
    elm=[]
    lat=[]

    r1=np.array((3.5,0.5,0))
    r2=np.array((0.5,3.5,0))

    l1=np.array((4,0,0))
    l2=np.array((0,4,0))
    l3=np.array((0,0,100))

    elm1="Ga"
    elm2="Al"

    xyz.append(r1)
    xyz.append(r2)

    lat.append(l1)
    lat.append(l2)
    lat.append(l3)

    elm.append(elm1)
    elm.append(elm2)

    CM=load_data.make_correlation_matrix_fast(xyz,lat,elm)
    same=True
    for i in range(2):
        for j in range(2):
            if (np.abs(CM_manual[i,j]-CM[i,j])>0.001):
                same=False
    
    assert same

    # 7)                                                                                                          
    xyz=[]
    elm=[]
    lat=[]

    r1=np.array((0,0.5,0.5))
    r2=np.array((0,3.5,3.5))

    l1=np.array((100,0,0))
    l2=np.array((0,4,0))
    l3=np.array((0,0,4))

    elm1="Ga"
    elm2="Al"

    xyz.append(r1)
    xyz.append(r2)

    lat.append(l1)
    lat.append(l2)
    lat.append(l3)

    elm.append(elm1)
    elm.append(elm2)

    CM=load_data.make_correlation_matrix_fast(xyz,lat,elm)
    same=True
    for i in range(2):
        for j in range(2):
            if (np.abs(CM_manual[i,j]-CM[i,j])>0.001):
                same=False
    
    assert same

    # 8)       

    xyz=[]
    elm=[]
    lat=[]

    r1=np.array((0,0.5,3.5))
    r2=np.array((0,3.5,0.5))

    l1=np.array((100,0,0))
    l2=np.array((0,4,0))
    l3=np.array((0,0,4))

    elm1="Ga"
    elm2="Al"

    xyz.append(r1)
    xyz.append(r2)

    lat.append(l1)
    lat.append(l2)
    lat.append(l3)

    elm.append(elm1)
    elm.append(elm2)

    CM=load_data.make_correlation_matrix_fast(xyz,lat,elm)
    same=True
    for i in range(2):
        for j in range(2):
            if (np.abs(CM_manual[i,j]-CM[i,j])>0.001):
                print(CM_manual)
                print(CM)
                same=False


    assert same

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








