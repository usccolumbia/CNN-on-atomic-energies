import numpy as np
import matplotlib.pyplot as plt
import sys
'''
 Script to plot weights (of at least fully connected layers)
Usage:
-L i: plot i:th layer
-LT "FC" or "CL : specify layer type
-I i: which nodes? Not sure how this works
'''

cmd = sys.argv[1:]

# Parse command line arguments (in a horrible way)
ind = 0
layer_speficied = False
indices_specified=False
layer_type_specified=False
plot_density=100
for word in cmd:
    if word[0]=="-":
        # Specify layer
        if(word[1:]=="layer" or word[1:]=="L"): 
           il=int(cmd[ind+1])
           layer_specified=True
        # Specify indices (channels or nodes)
        elif(word[1:]=="indices" or word[1:]=="I"):
            indices_found=True
            ind2=1
            indices=[]
            while((ind+ind2)<len(cmd)) and not(cmd[ind+ind2][0]=="-"):
                split=cmd[ind+ind2].split(":")
                if len(split)>1:
                    for i in range(int(split[0]),int(split[1])+1):
                        indices.append(i)
                else:
                    indices.append(int(split[0]))
                ind2=ind2+1
        # Specify layer type
        elif(word[1:]=="layer_type" or word[1:]=="LT"):
            if (cmd[ind+1]=="CL"):
                layer_type_specified=True
                layer_type = "conv"
            elif(cmd[ind+1]=="FC"):
                layer_type_specified=True
                layer_type = "fc"
        elif(word[1:]=="speed" )or(word[1:]=="S"):
            plot_density=int(cmd[ind+1])

    ind=ind+1 # Ends: if word[0]=="-"


if(not(layer_type_specified)):
    print("Error: layer type unspecified")    
if(not(layer_specified)):
    print("Error: Layer index not specified")
    
w = np.load('weights_'+layer_type+'layer'+str(il)+'.npy')
b = np.load('biases_'+layer_type+'layer'+str(il)+'.npy')

if(not(indices_specified)):
        print("No indices specified, plotting everything")
        
        
if(layer_type=="fc"):
    Niter=len(w)
    Nin=w[0].shape[0]
    Nout=w[0].shape[0]

    if (not(indices_specified)):
        indices=range(w[0].shape[1])

    Nnode=len(indices)
    
    plt.ion()
    fig,ax=plt.subplots(2,1)
    
    for i in range(Niter):
        if(i%plot_density==0):
            ax[0].clear()
            ax[1].clear()
            ax[0].set_title("Iter"+str(i)+'/'+str(Niter))
            ax[0].set_ylabel("N_in")
            ax[1].set_ylabel("Bias value")
            ax[1].set_xlabel("Node index")
            ax[0].imshow(w[i])
            ax[1].plot(range(Nnode),b[i],'-*')
            fig.canvas.draw()
            plt.pause(0.0001)

    
