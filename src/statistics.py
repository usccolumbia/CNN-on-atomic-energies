import numpy as np
import theano
import theano.tensor as T
import hyppar

# Weights  and biases of 3 convlayers and on FC layer,
# to be saved during training
w=[]
b=[]


# Activations of the convlayers, to be saved during training
conv_out = []
fc_out   = []

def writeParameters(dir="output"):
    global w
    global b

    # Use: hyppar module
    Nl           = hyppar.NCL
    NFC          = hyppar.NFC
    Nc           = hyppar.Nchannel
    filter       = hyppar.filter
    image_spec_x = hyppar.image_spec_x
    image_spec_y = hyppar.image_spec_y
    fc_out       = hyppar.fc_out
    
    # Derived variables
    Niter   = len(w)
    
    for i in range(Nl):
        wim = np.zeros((Niter,Nc[i+1],Nc[i],filter[i][0],filter[i][1]))
        bim = np.zeros((Niter,Nc[i+1]))
        for j in range(Niter):
            wim[i,:,:,:,:] = w[j][i]
            bim[i,:]       = b[j][i]
        np.save(dir+'/weights_convlayer_'+str(i),wim)
        np.save(dir+'/biases_convlayer_'+str(i),bim)


    if (NFC==0):
        return

    if(Nl>0):
        num_in = Nc[NCL]*image_spec_x[-1]*image_spec_y[-1]
    else:
        num_in = image_spec_x[0]*image_spec_y[0]


    ind=0
    num_out = fc_out[ind] 
    for i in range(Nl,Nl+NFC):

        wfc=np.zeros((Niter,num_in,num_out))
        for j in range(Niter):
            wfc[j,:,:]=w[j][i]
        bfc=np.zeros((Niter,num_out))
        for j in range(Niter):
            bfc[j,:]=b[j][i]
        np.save(dir+'/weights_fclayer'+str(ind),wfc)
        np.save(dir+'/biases_fclayer'+str(ind),bfc)
        ind=ind+1
        num_in=num_out
        
        num_out=fc_out[ind]

def writeActivations(dir="output"):
    global conv_out

    # Use: hyppar module
    Nl           = hyppar.NCL
    Nc           = hyppar.Nchannel
    image_spec_x = hyppar.image_spec_x
    image_spec_y = hyppar.image_spec_y

    # Derived variables
    Niter   = len(conv_out)
    Nsample = len(conv_out[0])
    
    for i in range(Nl):
        for j in range(Nsample):
            image = np.zeros((Niter,Nc[i+1],image_spec_x[i+1], image_spec_y[i+1]))
            for k in range(Niter):
                image[k,:,:,:] = conv_out[k][j][i] 
            np.save(dir+'/activations_layer'+str(i)+'_sample'+str(j),image)
        
def writefcActivations(dir="output"):
    global fc_out
    NFC = hyppar.NFC

    Niter = len(fc_out)
    Nlayer = len(fc_out[0])
    Nsample = hyppar.Nsamples_fc

    for i in range(NFC):
        Nnode=hyppar.fc_out[i]
        image = np.zeros((Niter,Nnode,Nsample))
        for j in range(Nnode):
            for k in range(Niter):
                image[k,j,:] = fc_out[k][i][j]
        np.save(dir+'/activations_fclayer'+str(i),image)

            
def saveActivations(activations):
    '''
    Completely saves the current activation tensors of
    the convolutional layers from 2 random input samples.

    Data structure for saved activations:
    conv_out   : #iter x [#samples x [#Layers x #channels x xdim x ydim]]
   
    Example: 
    iter i, sample s, layer l has a numpy entry obtained by
    conv_out[i][s][l] := Nlayers X xdim X ydim
    '''
    global conv_out

    iter=[] # Conv_out elements

    sample1=[] # iter elements
    sample2=[]

    for i in range(len(activations)):
        A1=np.array(activations[i][0]) # layer i, sample 1
        A2=np.array(activations[i][1]) # layer i, sample 2
        sample1.append(A1)
        sample2.append(A2)#

    iter.append(sample1)
    iter.append(sample2)
    
    conv_out.append(iter)
    
def savefcActivations(activations):
    '''
    Completely saves the current activation tensors of 
    the fully connected layers from input samples. 
    Data structure for saved activations: 
    fc_out   : #iter x [#layers x [#samples  x #nodes]] 
    Example: 
    iter i, layer l, node n has a numpy entry obtained by 
    fc_out[i][l][n] := Nsamples
    '''
    global fc_out

    Nl = len(activations)
    Ns = hyppar.Nsamples_fc

    
    iter=[] # fc_out snapshot
    
    for i in range(Nl):              # Layer
        layer = []
        A = np.array(activations[i])
        for j in range(hyppar.fc_out[i]):   # Node
            node = []
            for k in range(Ns):      # feature
                node.append(A[k,j])
            layer.append(node)
        iter.append(layer)
            
    fc_out.append(iter)

def saveParameters(params):
    ''' 
    Takes a snapshot of all of the current weights and biases
    '''
    global w 
    global b


    snapw = []
    snapb = []
    for i in range(len(params)):
        if (i%2==0):
            snapw.append(params[i].get_value())
        else:
            snapb.append(params[i].get_value())

    w.append(snapw)
    b.append(snapb)

    
