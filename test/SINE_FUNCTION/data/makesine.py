import numpy as np
import matplotlib.pyplot as plt

x=np.zeros((len(np.arange(0,2*3.141,0.01)),1))
N=len(x)
x[:,0]=np.arange(0,2*3.141,0.01)
x3=np.zeros((3*N,1))
for i in range(N):
    x3[i,0]     =x[i]
    x3[i+N,0]   =x[i]
    x3[i+2*N,0] =x[i]
print(3*x.shape[0])
y=np.sin(x3)
np.savetxt('Ydata',y)

for i in range(len(x3)):
    np.savetxt('Xdata/'+str(i),x3[i])
plt.plot(x3[2*N:3*N,0],y[2*N:3*N,0])
plt.show()
