### Part 1 initial dataset analysis
# * Importing datasets
# * Feature analysis
# * Normalisation and Standardization
# * Training & testing without autoencoder (?)

from scipy.io import arff
import pandas as pd

def main():
    nsl()

### NSL-KDD
def nsl():
    nsldata = arff.loadarff('data/nsl/KDDTrain+.arff')
    nsldf = pd.DataFrame(nsldata[0])
    print(nsldf.columns)

### Microsoft Malware
def malware():
    print('TODO')

if __name__ == '__main__':
    main()
