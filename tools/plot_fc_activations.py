import numpy as np
import matplotlib.pyplot as plt
import sys

'''
A script tp plot FC activations. 

- Currently this probably works only for 0-dimensional data.
- First command line argument is the FC layer index, further 
  arguments specify which nodes to plot.
'''

# Density of the iteration plots
plot_period=1

Narg = len(sys.argv)

if Narg>1:
    fclayer=int(sys.argv[1])
else:
    print("Which layer?")
plot_true_dist=False
for word in sys.argv:
    if word=="-PV":
        plot_true_dist=True

A0 = np.load('activations_fclayer'+str(fclayer)+'.npy')

Xdata=np.load('validation_features.npy') 

Ydata=np.load('validation_targets.npy') 

# Specify which nodes to plot
nodes=[]
if(Narg>3):
    idx=0
    for i in range(3,Narg):
        nodes.append(int(sys.argv[i]))
else:
    nodes=np.arange(len(A0[0,:,0]))


Niter=len(A0)
Nnodes=len(nodes)
plt.ion()
fig,ax=plt.subplots(1)

for i in range(Niter):
    if(i%plot_period==0):
        ax.clear()
        ax.set_title('Iteration '+str(i)+'/'+str(Niter))
        for j in range(Nnodes):
            ax.plot(A0[i,nodes[j],:],'b-')
        if(plot_true_dist):ax.plot(Ydata[:,0],'r-')
        fig.canvas.draw()
        plt.pause(0.0002)
    
