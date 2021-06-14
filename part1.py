### Part 1 initial dataset analysis
# * Importing datasets
# * Feature analysis
# * Normalisation and Standardization
# * Training & testing without autoencoder (?)

from scipy.io import arff
import pandas as pd

def main():
    print('Test')

### NSL-KDD
def nsl():
    nsldata = arff.loadarff('../data/nsl/KDDTrain+.arff')

### Microsoft Malware
def malware():
    print('TODO')

if __name__ == '__main__':
    main()
