import numpy as np
import theano
import theano.tensor as T

a=np.random.normal(0,0.0001,(2,2))
b=np.zeros((2,))
print(a)
print(b)
print('\n')

A=theano.shared(np.asarray(a),borrow=True)
B=theano.shared(np.asarray(b),borrow=True)
print(A.get_value())
print(B.get_value())
print('\n')
print('\n')



c=np.random.normal(10,0.00001,(2,2))
d=np.ones((2,))
print(c)
print(d)
print('\n')

C=theano.shared(np.asarray(c),borrow=True)
D=theano.shared(np.asarray(d),borrow=True)


print(C.get_value())
print(D.get_value())
print('\n')
print('\n')
print('\n')
print('\n')

AA=[A,B]

print(AA[0].get_value())

CC=[C,D]

print(CC[1].get_value())

K=AA+CC
print(K)

print(A)
print(K[0].get_value())
