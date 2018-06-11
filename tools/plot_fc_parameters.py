import numpy as np
import random as rd
import matplotlib.pyplot as plt
import sys


'''
A script to plot a random set of weights and biases of a fully connected layer.

- Specify layer and number of samples in input.

- Weakness: Random samples may produce duoplicates, therefore Nsamples is not
  always the correct number. 
'''

rd.seed(23345)

cmd = sys.argv[1:]

ind = 0
layer = 0
Nsamples=99999999999999999999

# Parse command line arguments
for word in cmd:
    if word[0]=="-":
        vipu = word[1:]
        if(vipu=="layer")   : layer=int(cmd[ind+1])
        if(vipu=="samples") : Nsamples=int(cmd[ind+1])
    ind=ind+1    

# Load weights and biases
w=np.load('weights_fclayer'+str(layer)+'.npy')
b=np.load('biases_fclayer'+str(layer)+'.npy')

# Print info on the command line
print("\n Plotting weights of fully connected layer "+str(layer)+'. \n')
print("Shape of the layer:")
print(w[0].shape)
xdim=w[0].shape[0]
ydim=w[0].shape[1]
print("\n")
if(Nsamples<w[0].size):
    print(str(Nsamples)+" random samples will be plotted.")
else:
    print("All of the parameters will be plotted.")
    Nsamples=w[0].size

# Plot
fig,(ax1,ax2)=plt.subplots(2)
ax1.set_title("Parameter plot of FC layer "+str(layer))
for i in range(Nsamples):
    ix=np.random.randint(xdim)
    iy=np.random.randint(ydim)
    ax1.plot(w[:,ix,iy])
    ax2.plot(b[:,iy])
    ax1.set_ylabel("Weights")
    ax2.set_ylabel("Biases")
    ax2.set_xlabel("Iteration step")
         
plt.show()
