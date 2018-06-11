import numpy as np
import random as rd
import matplotlib.pyplot as plt
import sys

A=np.load('activations_layer0_sample0.npy')

Niter=A.shape[0]
plt.ion()
fig,ax=plt.subplots(2,2)
for i in range(Niter):
    ax[0,0].clear()
    ax[1,0].clear()
    ax[0,1].clear()
    ax[1,1].clear()
    ax[0,0].imshow(A[i,0,:,:])
    ax[1,0].imshow(A[i,1,:,:])
    ax[0,1].imshow(A[i,2,:,:])
    ax[1,1].imshow(A[i,3,:,:])
    ax[0,0].set_title('Iteration '+str(i)+'/'+str(Niter))
    fig.canvas.draw()
    plt.pause(0.1)
