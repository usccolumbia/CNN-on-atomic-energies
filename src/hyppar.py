
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
#####################################

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

    print('\n STRUCTURE OF NETWORK')
    print('\n Number of channels :')
    for i in Nchannel:
        print(i)

    print('\n REGULARIZATION')
    print('\n Regularization : '+str(reg))


