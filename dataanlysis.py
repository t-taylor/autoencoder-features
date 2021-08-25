### Dataset analysis
# Looking at the qualities of each of the datasets.

import part1 as dss
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import collections as clls

def main():
    nslmul = dss.nsl_multiclass()
    nslbin = dss.multi_to_bin(nslmul)
    print(nslmul[0]['class'].value_counts())
    print(nslmul[0].shape)
    print(nslmul[1]['class'].value_counts())
    print(nslmul[1].shape)
    print('col size', len(dss._txtheaders))

    #fig, ax = plt.subplots()
    #sns.countplot(ax=ax,data=nslmul[0], x='class')
    #plt.show()
    print(get_col_types())

def get_col_types():
    nsldf_train = pd.read_csv('data/nsl/KDDTrain+.txt', names=dss._txtheaders)

    nsldf_test = pd.read_csv('data/nsl/KDDTest+.txt', names=dss._txtheaders)

    # Normalise & Standardise
    train_norm = pd.DataFrame()
    test_norm = pd.DataFrame()
    freq = clls.defaultdict(int)
    for k, v in nsldf_train.dtypes.items():
        freq[v] += 1
    print(freq)

if __name__ == '__main__':
    main()
