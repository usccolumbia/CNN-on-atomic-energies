# CNN-on-atomic-energies
Neural network approach for predicting crystal formation energies

THE CODE:

src:
- All the source code is here
- "test"-directory (to be named "output") contains all output of the training and predictions on the test set.

EXECUTION DIRECTORY MUST HAVE THESE:

CM:
- Coulomb matrices for training and test sets, to be loaded by "main.py" in file src.

data:
- All raw data for constructing coulomb matrices, loading energies for formation and band
  gaps and more.
  
  
Train CNN by going to src and typing: python main.py
