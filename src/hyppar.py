
###### DEFAULT VALUES: ###############
datapath = "/wrk/krsimula/DONOTREMOVE/NEURAL_NETWORKS/CNN-on-atomic-energies/"
# Number of datapoints
Ndata = 2200
# Number of aug points
Naug = 2400
# Training set size
Ntrain = 3800
# Validation set size
Nval = 800
# Learning_rate
learning_rate = 0.05
# Number of epochs
Nepoch = 10
# Minibatch size
mbs = 20
# Regularization
reg = 0.001

######### Structural parameters #################
# 2D input
in_x = 80
in_y = 80
in_z = 1 
# Number of convolutional layers
NCL = 3
# Number of chanels
Nchannel        =  [1, 8, 8, 8]
# Filters
filter1         = [9, 9]
filter2         = [5, 5]
filter3         = [5, 5]
filter4         = [5, 5] 
filter5         = [5, 5] 
filter6         = [5, 5] 
filter7         = [5, 5] 

# Pooling
pool1          = [2, 2]
pool2          = [2, 2]
pool3          = [2, 2]
pool4          = [2, 2]
pool5          = [2, 2]
pool6          = [2, 2]
pool7          = [2, 2]

# Activations  
activation1    = 'tanh'
activation2    = 'tanh'
activation3    = 'tanh'
activation4    = 'tanh'
activation5    = 'tanh'
activation6    = 'tanh'
activation7    = 'tanh' 

# Number of fully connected layers
NFC            = 1
# Activations
fc_activation  = 'lin'
# Ignore border in pooling
ignore_border  = False

#####################################

def setStructureParameters():
    '''
    Defines the parameters of how the data is
    shaped in CNN. All the defined parameters
    depend on the input file parameters, so this
    should be invisible to user. 
    '''
    # Use: local module
    global NCL
    global NFC
    global in_x
    global in_y
    global filter1
    global filter2
    global filter3
    global pool1
    global pool2
    global pool3
    global ignore_border
    global activation1
    global activation2
    global activation3
    global fc_activation
    global Nchannel
    
    # New global parameters:
    global image_spec_x
    global image_spec_y
    
    global pool
    pool = []
    pool.append(pool1)
    pool.append(pool2)
    pool.append(pool3)
    pool.append(pool4)
    pool.append(pool5)
    pool.append(pool6)
    pool.append(pool7)

    global filter
    filter = []
    filter.append(filter1)
    filter.append(filter2)
    filter.append(filter3)
    filter.append(filter4)
    filter.append(filter5)
    filter.append(filter6)
    filter.append(filter7)

    global activation
    activation = []
    activation.append(activation1)
    activation.append(activation2)
    activation.append(activation3)
    activation.append(activation4)
    activation.append(activation5)
    activation.append(activation6)
    activation.append(activation7)


    image_spec_x = []
    image_spec_y = []

    image_spec_x.append(in_x)
    image_spec_y.append(in_y)

    for i in range(1,NCL+1):
        image_spec_x.append(image_spec_x[i-1]-filter[i-1][0]+1)
        image_spec_y.append(image_spec_y[i-1]-filter[i-1][1]+1)
        if (image_spec_x[i] < 1 or image_spec_y[i] < 1):
            print("\n ERROR!!! Too large filter in convlayer "+str(i)+" \n")
        if (ignore_border):
            image_spec_x[i]=(image_spec_x[i]-image_spec_x[i]%pool[i-1][0])/pool[i-1][0]
            image_spec_y[i]=(image_spec_y[i]-image_spec_y[i]%pool[i-1][1])/pool[i-1][1]
        else:
            image_spec_x[i]=(image_spec_x[i]+image_spec_x[i]%pool[i-1][0])/pool[i-1][0]
            image_spec_y[i]=(image_spec_y[i]+image_spec_y[i]%pool[i-1][1])/pool[i-1][1]

    print('\n STRUCTURE OF NETWORK')
    print('\n Number of convolutional layers      : '+str(NCL))
    print(' Number of fully connected layers    : '+str(NFC))

    print('\n Shape of the input:')
    print(in_x,in_y)
    print('\n')

    for i in range(NCL):    
    
        print("*** Convlayer "+ str(i+1)+" ***")
        print('Filter shape:') # Generalize!!!!!!!!!!!!                                                  
        print(filter[i][0],filter[i][1])
        print(' Pooling:')
        print(pool[i][0],pool[i][1])
        print(' Activation: ')
        print(activation[i])
        print('Output image1:')
        print(Nchannel[i],image_spec_x[i+1],image_spec_y[i+1])
        print('\n')
        
    print(' Fully connected layer activation   : '+fc_activation)
        

# parse: look value for keyword from input
def parse(filename,varname):
    '''
    Expected to find input file in format:
    varname   :   #varvalue

    Does not return anything if not found, but prints warning
    '''
    var=[]
    with open(filename) as f:
        input=f.readlines()
        for line in input:
            words=line.split()
            if len(words)>0:
                if(words[0].lower() == varname.lower()):
                    var=words[2:]
    if len(var)>0:
        return var
    else:
        print("Warning: Value for variable " + varname + " not found from input. Using default value.")
        return var

def setInput(filename='input'):
    global datapath
    global Ndata
    global Naug
    global Ntrain
    global Nval
    global learning_rate
    global Nepoch
    global mbs
    global Nchannel
    global reg
    global in_x
    global in_y
    global NCL
    global Nchannel
    global filter1
    global filter2
    global filter3
    global pool1
    global pool2
    global pool3
    global activation1
    global activation2
    global activation3
    global NFC
    global fc_activation
    global ignore_border

    # Declare (again) the default values
    learning_rate = 0.05
    Nepoch = 10
    mbs = 20
    reg = 0.001
    
    in_x = 80
    in_y = 80
    in_z = 1
    NCL = 3
    Nchannel        =  [1, 8, 8, 8]
    filter1         = [9, 9]
    filter2         = [5, 5]
    filter3         = [5, 5]
    filter4         = [5, 5]
    filter5         = [5, 5]
    filter6         = [5, 5]
    filter7         = [5, 5]
    
    pool1          = [2, 2]
    pool2          = [2, 2]
    pool3          = [2, 2]
    pool4          = [2, 2]
    pool5          = [2, 2]
    pool6          = [2, 2]
    pool7          = [2, 2]
    
    activation1    = 'tanh'
    activation2    = 'tanh'
    activation3    = 'tanh'
    activation4    = 'tanh'
    activation5    = 'tanh'
    activation6    = 'tanh'
    activation7    = 'tanh'
    
    NFC            = 1
    fc_activation  = 'lin'
    
    
    
    datapath_buffer = parse(filename,'datapath')
    if len(datapath_buffer)>0:
        datapath = datapath_buffer[0]
    
    Ndata_buffer = parse(filename,'Ndata')
    if len(Ndata_buffer) > 0:
        Ndata=int(Ndata_buffer[0])
    
    Naug_buffer = parse(filename,'Naug')
    if len(Naug_buffer) > 0:
        Naug = int(Naug_buffer[0])
    
    Ntrain_buffer = parse(filename,'Ntrain')
    if len(Ntrain_buffer) > 0:
        Ntrain = int(Ntrain_buffer[0])
    
    Nval_buffer = parse(filename,'Nval')
    if len(Nval_buffer) > 0:
        Nval = int(Nval_buffer[0])
    
    learning_rate_buffer = parse(filename,'learning_rate')
    if len(learning_rate_buffer) > 0:
        learning_rate        = float(learning_rate_buffer[0])

    Nepoch_buffer = parse(filename,'Nepoch')
    if len(Nepoch_buffer) > 0:
        Nepoch        = int(Nepoch_buffer[0])
    
    mbs_buffer    = parse(filename,'mbs')
    if len(mbs_buffer) > 0:
        mbs           = int(mbs_buffer[0])

    reg_buffer      = parse(filename,'reg')
    if len(reg_buffer) > 0:
        reg             = float(reg_buffer[0])

    buffer = parse(filename,'in_x')
    if len(buffer) > 0:
        in_x = int(buffer[0])
    
    buffer = parse(filename,'in_y')
    if len(buffer) > 0:
        in_y = int(buffer[0])
        
    buffer = parse(filename,'in_z')
    if len(buffer) > 0:
        in_z = int(buffer[0])
    
    buffer = parse(filename,'NCL')
    if len(buffer) > 0:
        NCL = int(buffer[0])

    buffer = parse(filename,'Nchannel')
    if len(buffer) > 0:
        Nchannel[0] = in_z
        for i in range(1,NCL+1):
            Nchannel[i] = int(buffer[i-1])

    buffer = parse(filename,'filter1')
    if len(buffer) > 0:
        for i in range(2): # ! This can be easily generalized to 1 or 3D
            filter1[i] = int(buffer[i])

    buffer = parse(filename,'filter2')
    if len(buffer) > 0:
        for i in range(2): # ! This can be easily generalized to 1 or 3D                                         
             filter2[i] = int(buffer[i])

    buffer = parse(filename,'filter3')
    if len(buffer) > 0:
        for i in range(2): # ! This can be easily generalized to 1 or 3D                                         
            filter3[i] = int(buffer[i])

    buffer = parse(filename,'filter4')
    if len(buffer) > 0:
        for i in range(2): # ! This can be easily generalized to 1 or 3D
            filter4[i] = int(buffer[i])

    buffer = parse(filename,'filter5')
    if len(buffer) > 0:
        for i in range(2): # ! This can be easily generalized to 1 or 3D                                         
             filter5[i] = int(buffer[i])

    buffer = parse(filename,'filter6')
    if len(buffer) > 0:
        for i in range(2): # ! This can be easily generalized to 1 or 3D                                         
            filter6[i] = int(buffer[i])

    buffer = parse(filename,'filter7')
    if len(buffer) > 0:
        for i in range(2): # ! This can be easily generalized to 1 or 3D                                         
            filter7[i] = int(buffer[i])

    buffer = parse(filename,'pool1')
    if len(buffer) > 0:
        for i in range(2): # ! This can be easily generalized to 1 or 3D                                         
            pool1[i] = int(buffer[i])

    buffer = parse(filename,'pool2')
    if len(buffer) > 0:
        for i in range(2): # ! This can be easily generalized to 1 or 3D                                         
            pool2[i] = int(buffer[i])

    buffer = parse(filename,'pool3')
    if len(buffer) > 0:
        for i in range(2): # ! This can be easily generalized to 1 or 3D                                         
            pool3[i] = int(buffer[i])

    buffer = parse(filename,'pool4')
    if len(buffer) > 0:
        for i in range(2): # ! This can be easily generalized to 1 or 3D                                         
            pool4[i] = int(buffer[i])
    
    buffer = parse(filename,'pool5')
    if len(buffer) > 0:
        for i in range(2): # ! This can be easily generalized to 1 or 3D                                         
            pool5[i] = int(buffer[i])

    buffer = parse(filename,'pool6')
    if len(buffer) > 0:
        for i in range(2): # ! This can be easily generalized to 1 or 3D                                         
            pool6[i] = int(buffer[i])
    
    buffer = parse(filename,'pool7')
    if len(buffer) > 0:
        for i in range(2): # ! This can be easily generalized to 1 or 3D                                         
            pool7[i] = int(buffer[i])


    buffer = parse(filename,'activation1')
    if len(buffer) > 0:
        activation1=buffer[0]

    buffer = parse(filename,'activation2')
    if len(buffer) > 0:
        activation2=buffer[0]

    buffer = parse(filename,'activation3')
    if len(buffer) > 0:
        activation3=buffer[0]

    buffer = parse(filename,'activation4')
    if len(buffer) > 0:
        activation4=buffer[0]

    buffer = parse(filename,'activation5')
    if len(buffer) > 0:
        activation5=buffer[0]

    buffer = parse(filename,'activation6')
    if len(buffer) > 0:
        activation6=buffer[0]

    buffer = parse(filename,'activation7')
    if len(buffer) > 0:
        activation7=buffer[0]


    buffer = parse(filename,'fc_activation')
    if len(buffer) > 0:
        fc_activation=buffer[0]

    buffer = parse(filename,'ignore_border')
    if (len(buffer) > 0):
        ignore_border=buffer[0]

    print("\n *** Reading input: Done ***")
    
    print("\n Data is read from directory: "+datapath)
    
    print("\n DATASET PARAMETERS:")
    
    print('\n Number of datapoints                : '+str(Ndata) )
    print(' Number of augmentation points       : '+str(Naug))
    print(' Number of total data                : '+str(Ndata+Naug))
    print('\n Number of training points           : '+str(Ntrain))
    print(' Number of validation points         : '+str(Nval))

    print('\n OPTIMIZATION PARAMETERS:')

    print('\n Learning rate                       : '+str(learning_rate))
    print(' Number of epochs                    : '+str(Nepoch))
    print(' Minibatch size                      : '+str(mbs))

    print('\n REGULARIZATION')
    print('\n Regularization : '+str(reg))


