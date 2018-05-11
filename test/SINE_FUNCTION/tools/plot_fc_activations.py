import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv)>1:
    fclayer=int(sys.argv[1])
else:
    print("Which layer?")
if len(sys.argv)>2:
    node=int(sys.argv[2])

A0 = np.load('activations_fclayer'+str(fclayer)+'.npy')

Niter=len(A0)
plt.ion()
fig,ax=plt.subplots(1)

for i in range(Niter):
    if(i%100==0):
        ax.clear()
        ax.set_title('Iteration '+str(i)+'/'+str(Niter))
        ax.plot(A0[i,node,:])
        fig.canvas.draw()
        plt.pause(0.0002)
    
