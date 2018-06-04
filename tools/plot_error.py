import numpy as np
import matplotlib.pyplot as plt

Et = np.load('train_error.npy')
Ev = np.load('valid_error.npy')

Niter = len(Et)

print(Et[6])

fig,ax = plt.subplots(2,sharex=True)
ax[0].grid()
ax[1].grid()
ax[0].plot(Et,'r-')
ax[1].plot(Ev,'b-')
ax[0].set_ylabel('Training error')
ax[1].set_xlabel('Iteration step')
ax[1].set_ylabel('Validation error')
plt.show()

