import numpy as np
import matplotlib.pyplot as plt
import sys


a=np.load('biases_convlayer_1.npy')
print(a.shape)
for i in range(8):
    for j in range(9):
        print(a[i,j])
