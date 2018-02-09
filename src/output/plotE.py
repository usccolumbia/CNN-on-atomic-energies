import numpy as np
import matplotlib.pyplot as plt
import sys

target=np.loadtxt('E_target.txt')

fig=1
print(len(sys.argv))
for i in range(1,len(sys.argv)):
    print(sys.argv[i])
    pred=np.loadtxt('E_pred_'+str(sys.argv[i])+'.txt')
    plt.figure(fig); fig=fig+1
    plt.plot(target,pred,'r+')
    plt.plot(target,target,'b-')
    plt.title('Iter '+str(sys.argv[i]))
plt.show()
