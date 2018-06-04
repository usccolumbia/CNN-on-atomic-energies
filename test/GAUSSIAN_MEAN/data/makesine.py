import numpy as np
import matplotlib.pyplot as plt

def gaussian(x,y):
    return np.exp(-(x**2+y**2)/10)

x = np.zeros((len(np.arange(-0.5,0.5,0.002)),1)) # Datapoints
y = np.zeros((len(np.arange(-0.5,0.6,0.1)),1))     # Features
x[:,0]=np.arange(-0.5,0.5,0.002)
y[:,0]=np.arange(-0.5,0.6,0.1)
Nx=len(x)
Ny=len(y)

grid = np.zeros((Nx,Ny))

for i in range(Nx):
    for j in range(Ny):
        grid[i,j] = gaussian(x[i],y[j])

print(grid.shape)
        
labels = np.zeros((3*Nx,1))
for i in range(Nx):
    a = np.mean(grid[i,:])
    labels[i,0]      = a
    labels[i+Nx,0]   = a
    labels[i+2*Nx,0] = a

np.savetxt('Ydata',labels)


for i in range(Nx):
    XX = np.zeros((Ny,1))
    XX[:,0] = grid[i,:]

    np.save('Xdata/'+str(i),XX)
    np.save('Xdata/'+str(i+Nx),XX)
    np.save('Xdata/'+str(i+2*Nx),XX)
    
