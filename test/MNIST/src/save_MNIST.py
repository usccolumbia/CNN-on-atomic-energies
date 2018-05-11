import theano
import theano.tensor as T
import numpy as np
import six.moves.cPickle as pickle
import gzip
import os
import random as rd
import matplotlib.pyplot as plt

dataset = 'mnist.pkl.gz'
data_dir, data_file = os.path.split(dataset)
rd.seed(23455)
# Check if data file present
if data_dir == "" and not os.path.isfile(dataset):
    new_path = os.path.join('', dataset)
    if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
        dataset = new_path

# Download the file from MILA if not present                                                                 
if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
    from six.moves import urllib
    origin = (
        'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    )
    print('Downloading data from %s' % origin)
    urllib.request.urlretrieve(origin, dataset)
    
print('***** Loading data *****')
# Open the file                                                                                              
with gzip.open(dataset, 'rb') as f:
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    except:
        train_set, valid_set, test_set = pickle.load(f)

train_x,train_y=train_set
val_x,val_y=valid_set
test_x,test_y=test_set

print(train_x.shape)
print(val_x.shape)
print(test_x.shape)

Ntrain=train_x.shape[0]
Nval=val_x.shape[0]
Ntest=test_x.shape[0]

for i in range(Ntrain):
    X=train_x[i].reshape(28,28)
    np.savetxt('Xdata/'+str(i),X)

np.savetxt('Ydata',y_train)

#X=train_x[0]
#plt.imshow(X.reshape(28,28))
#plt.show()
