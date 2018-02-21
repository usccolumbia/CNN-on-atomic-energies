import numpy as np
import matplotlib.pyplot as plt

w1=np.load('weights_convlayer1.txt.npy')
print(w1.shape)
Niter=w1.shape[0]
print(Niter)

plt.ion()
fig, ax = plt.subplots(4,2)
for i in range(Niter):
    step=0
    for j in range(4):
        for k in range(2):
            step=j*2+k
            ax[j,k].matshow(w1[i,step,0,:,:])
            fig.canvas.draw()
            plt.pause(0.5)
        
