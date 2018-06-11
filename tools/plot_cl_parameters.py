import numpy as np
import random as rd
import matplotlib.pyplot as plt
import sys

rd.seed(23345)

cmd = sys.argv[1:]

# Parse command line arguments (in a horrible way)
ind = 0
layer = 0
Nsamples=1
channels = []

for word in cmd:
    if word[0]=="-":
        vipu = word[1:]
        if(vipu=="layer")   : layer=int(cmd[ind+1])
        if(vipu=="samples") : Nsamples=int(cmd[ind+1])
        if(vipu=="channels"):
            cc=cmd[ind+1].split(":")
            channels=range(int(cc[0],int(cc[1])))
    ind=ind+1
    
w=np.load('weights_convlayer_'+str(layer)+'.npy')
b=np.load('biases_convlayer_'+str(layer)+'.npy')

if (len(channels)==0):channels=range(w[0].shape[0])
Nchannels=len(channels)

print('\n Plotting weights of the convolutional layer '+str(layer)+'. \n')
print('Shape of the filter: ')
print(w[0].shape)

if(channels[-1]>w[0].shape[0]):
    print('\n Not enough channels for your request! \n')
else:
    print('\n Channels to be plotted:')
    print(channels)
    print('\n From each channel, '+str(Nsamples)+' random weights will be plotted.')

xdim=w[0].shape[2]
ydim=w[0].shape[3]


fig,ax=plt.subplots(Nchannels,2)
ax[0,0].set_title("Weights")
ax[0,1].set_title("Biases")
for i in range(Nsamples):
    ix=np.random.randint(xdim)
    iy=np.random.randint(ydim)
    for j in range(Nchannels):
        ax[j,0].plot(w[:,j,0,ix,iy])
        ax[j,1].plot(b[:,j])
ax[Nchannels-1,0].set_xlabel("Iteration step")
ax[Nchannels-1,1].set_xlabel("Iteration step")

plt.show()

