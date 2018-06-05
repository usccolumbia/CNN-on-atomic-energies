import hyppar

# Is main being run from separate directory?
remote_run=False
# If remote_run=True, we need a folder
remote_folder=""

def parseArgs(args,path):
    global remote_run
    global remote_folder
    ind=0
    for i in args:
        ind=ind+1
        if i[0]=="-":
            if i[1:]=="remote_folder":
                remote_run=True
                remote_folder=args[ind]

    if remote_run:
        hyppar.current_dir=remote_folder
    else:
        hyppar.current_dir=path
        
