import numpy as np
import matplotlib.pyplot as plt

def load_data(e=False,w0=False,w1=False,w2=False,wf=False,pred=False):

    if(pred):
        E_pred=np.loadtxt('E_pred.txt',dtype='float')
        E_target=np.loadtxt('E_target.txt',dtype='float')

        return E_pred,E_target
        
    Nf=[]
    if(e):
        Et=np.loadtxt('et.txt',dtype='float')
        Ev=np.loadtxt('ev.txt',dtype='float')
    
        Niter=Et.shape[0]
        return Et,Ev

    if (w0):
        w01=np.loadtxt('w0_sample1.txt',dtype='float')
        w02=np.loadtxt('w0_sample2.txt',dtype='float')
        w03=np.loadtxt('w0_sample3.txt',dtype='float')
    
        Nf = w01.shape[1]
        return w01,w02,w03,Nf
    
    if(w1):
        w11=np.loadtxt('w1_sample1.txt',dtype='float')
        w12=np.loadtxt('w1_sample2.txt',dtype='float')
        w13=np.loadtxt('w1_sample3.txt',dtype='float')
    
        Nf=w11.shape[1]
        return w11,w12,w13,Nf

    if(w2):
        w21=np.loadtxt('w2_sample1.txt',dtype='float')
        w22=np.loadtxt('w2_sample2.txt',dtype='float')
        
        Nf = w21.shape[1]
        return w21,w22,Nf
    
    if(wf):
        wfc11=np.loadtxt('wfc1_sample1.txt')
        wfc12=np.loadtxt('wfc1_sample2.txt')
        wfc13=np.loadtxt('wfc1_sample3.txt')
        
#        wfc21=np.loadtxt('wfc2_sample1.txt')
#        wfc22=np.loadtxt('wfc2_sample2.txt')
#        wfc23=np.loadtxt('wfc2_sample3.txt')

        return wfc11,wfc12,wfc13

def plotpred():
    fig,ax=plt.subplots(1)
    Ep,Et=load_data(pred=True)
    ax.plot(Et,Ep,'r*')
    ax.set_xlabel('Target values'); ax.set_ylabel('Predicted values')
    xx=np.linspace(-0.5,np.amax(Et)+0.5,200)
#    print(xx)
    ax.plot(xx,xx)
    
def plotE():
    Et,Ev=load_data(e=True)
    fig_wf1, axwf = plt.subplots(2, sharex=True)

    axwf[0].plot(Et)
    axwf[0].set_title('Training error')
    axwf[1].plot(Ev)
    axwf[1].set_title('Validation error')
    


def printw0():
    w01,w02,w03,Nf=load_data(w0=True)
    fig_w0, axw0 = plt.subplots(Nf, sharex=True)
    for ch_count in range(Nf):
        l1 = axw0[ch_count].plot(w01[:, ch_count])
        l2 = axw0[ch_count].plot(w02[:, ch_count])
        l3 = axw0[ch_count].plot(w03[:, ch_count])
        axw0[ch_count].yaxis.set_label_position("right")
        axw0[ch_count].set_ylabel('Channel '+str(ch_count+1))

    axw0[0].set_title('Evolution of Three Arbitrary Weights in the channels of conv. layer 1')
    axw0[Nf-1].set_xlabel('Iterations')

def printw1():
    w11,w12,w13,Nf=load_data(w1=True)
    fig_w1, axw1 = plt.subplots(Nf, sharex=True)
    for ch_count in range(Nf):
        l1 = axw1[ch_count].plot(w11[:, ch_count])
        l2 = axw1[ch_count].plot(w12[:, ch_count])
        l3 = axw1[ch_count].plot(w13[:, ch_count])
        axw1[ch_count].yaxis.set_label_position("right")
        axw1[ch_count].set_ylabel('Channel '+str(ch_count+1))

    axw1[0].set_title('Evolution of Three Arbitrary Weightsin the channels of conv. layer 2')
    axw1[Nf-1].set_xlabel('Iterations')

def printw2():
    w21,w22,Nf=load_data(w2=True)
    fig_w2, axw2 = plt.subplots(Nf, sharex=True)
    for ch_count in range(Nf):
        l1 = axw2[ch_count].plot(w21[:, ch_count])
        l2 = axw2[ch_count].plot(w22[:, ch_count])
        
        axw2[ch_count].yaxis.set_label_position("right")
        axw2[ch_count].set_ylabel('Channel '+str(ch_count+1))

    axw2[0].set_title('Evolution of Three Arbitrary Weights in the channels of conv. layer 3')
    axw2[Nf-1].set_xlabel('Iterations')

def printwf():

    wfc11,wfc12,wfc13=load_data(wf=True)
    fig_wf1, axwf = plt.subplots(2, sharex=True)

    axwf[0].plot(wfc11)
    axwf[0].plot(wfc12)
    axwf[0].plot(wfc13)


 #   axwf[1].plot(wfc21)
 #   axwf[1].plot(wfc22)
 #   axwf[1].plot(wfc23)

    axwf[0].set_ylabel('FC layer 1')
  #  axwf[1].set_ylabel('FC layer 2')
    
    axwf[0].set_title('Evolution of Three Arbitrary Weights in the 9 Channels')
   # axwf[1].set_xlabel('Iterations')
                        

