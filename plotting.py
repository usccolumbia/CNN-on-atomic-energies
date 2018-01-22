import numpy as np
import matplotlib.pyplot as plt

def EnergyHist(Ef,color,name,xlabel):

    # the histogram of the data
    n, bins, patches = plt.hist(Ef, 6, facecolor=color, alpha=1)

    plt.xlabel(xlabel)
    plt.ylabel('Occurrences')
    plt.title(name)
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True)

    plt.show()
