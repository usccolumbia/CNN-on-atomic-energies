
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
# Number of channels
Nchannel = [8, 8, 8]
# Regularization
reg = 0.001

######### Structural parameters #################
# 2D input
in_x = 80
in_y = 80
# Number of convolutional layers
NCL = 3
# Number of chanels
Nchannel        =  [8, 8, 8]
# Filters
filter1         = [9, 9]
filter2         = [5, 5]
filter3         = [5, 5]
# Pooling
pool1          = [2, 2]
pool2          = [2, 2]
pool3          = [2, 2]
# Activations  
activation1    = 'tanh'
activation2    = 'tanh'
activation3    = 'tanh'
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

    if NCL > 0:
        # Layer 2 input size x
        global image_spec_2_x
        image_spec_2_x = in_x-filter1[0]+1
        if (ignore_border):
            image_spec_2_x=(image_spec_2_x-image_spec_2_x%pool1[0])/pool1[0]
        else:
            image_spec_2_x=(image_spec_2_x+image_spec_2_x%pool1[0])/pool1[0]
        # Layer 2 input size y
        global image_spec_2_y
        image_spec_2_y = in_y-filter1[1]+1
        if (ignore_border):
            image_spec_2_y=(image_spec_2_y-image_spec_2_y%pool1[1])/pool1[1]
        else:
            image_spec_2_y=(image_spec_2_y+image_spec_2_y%pool1[1])/pool1[1]

    if NCL > 1:
        # Layer 3 input size x                                                                                   
        global image_spec_3_x
        image_spec_3_x = image_spec_2_x-filter2[0]+1
        if (ignore_border):
            image_spec_3_x=(image_spec_3_x-image_spec_3_x%pool2[0])/pool2[0]
        else:
            image_spec_3_x=(image_spec_3_x+image_spec_3_x%pool2[0])/pool2[0]
        # Layer 3 input size y                                                                               
        global image_spec_3_y
        image_spec_3_y = image_spec_2_y-filter2[1]+1
        if (ignore_border):
            image_spec_3_y=(image_spec_3_y-image_spec_3_y%pool2[1])/pool2[1]
        else:
            image_spec_3_y=(image_spec_3_y+image_spec_3_y%pool2[1])/pool2[1]

    global fc_in_x
    global fc_in_y
    if NCL==1:
        fc_in_x=image_spec_2_x
        fc_in_y=image_spec_2_y
    elif NCL==2:
        fc_in_x=image_spec_3_x
        fc_in_y=image_spec_3_y
    elif NCL==3:
        fc_in_x = image_spec_3_x-filter3[0]+1
        if (ignore_border):
            fc_in_x=(fc_in_x-fc_in_x%pool3[0])/pool3[0]
        else:
            fc_in_x=(fc_in_x+fc_in_x%pool3[0])/pool3[0]
        fc_in_y = image_spec_3_y-filter3[1]+1
        if (ignore_border):
            fc_in_y=(fc_in_y-fc_in_y%pool3[1])/pool3[1]
        else:
            fc_in_y=(fc_in_y+fc_in_y%pool3[1])/pool3[1]


    print('\n STRUCTURE OF NETWORK')
    print('\n Number of convolutional layers      : '+str(NCL))
    print(' Number of fully connected layers    : '+str(NFC))
    
    print('\n Shape of the input:')
    print(in_x,in_y)
    print('\n')
    print("*** Convlayer 1: ***")
    print('Filter shape:') # Generalize!!!!!!!!!!!!                                                  
    print(filter1[0],filter1[1])
    print(' Pooling:')
    print(pool1[0],pool1[1])
    print(' Activation: ')
    print(activation1)
    print('Output image1:')
    print(Nchannel[0],image_spec_2_x,image_spec_2_y)
    print('\n')

    print("*** Convlayer 2: ***")
    print('\n Filter shape:')
    print(filter2[0],filter2[1])
    print(' Pooling:')
    print(pool2[0],pool2[1])
    print(' Activation:')
    print(activation2)
    print('Output image:')
    print(Nchannel[1],image_spec_3_x,image_spec_3_y)
    print('\n')

    print("*** Convlayer 3: ***")
    print('\n Filter shape')
    print(filter3[0],filter3[1])
    print(' Pooling.')
    print(pool3[0],pool3[1])
    print(' Activation: ')
    print(activation3)
    print('Output image:')
    print(Nchannel[2],fc_in_x,fc_in_y)

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

    Nchannel_buffer = parse(filename,'Nchannel')
    if len(Nchannel_buffer) > 0:
        for i in range(len(Nchannel)):
            Nchannel[i]    = int(Nchannel_buffer[i])

    reg_buffer      = parse(filename,'reg')
    if len(reg_buffer) > 0:
        reg             = float(reg_buffer[0])

    buffer = parse(filename,'in_x')
    if len(buffer) > 0:
        in_x = int(buffer[0])
    
    buffer = parse(filename,'in_y')
    if len(buffer) > 0:
        in_y = int(buffer[0])
    
    buffer = parse(filename,'NCL')
    if len(buffer) > 0:
        NCL = int(buffer[0])

    buffer = parse(filename,'Nchannel')
    if len(buffer) > 0:
        for i in range(NCL):
            Nchannel[i] = int(buffer[i])

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

    buffer = parse(filename,'activation1')
    if len(buffer) > 0:
        activation1=buffer[0]

    buffer = parse(filename,'activation2')
    if len(buffer) > 0:
        activation2=buffer[0]

    buffer = parse(filename,'activation3')
    if len(buffer) > 0:
        activation3=buffer[0]

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


