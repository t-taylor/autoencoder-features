### Part 1 initial dataset analysis
# * Importing datasets
# * Feature analysis
# * Normalisation and Standardization
# * Training & testing without autoencoder (?)

from pyparsing import Word, hexnums, WordEnd, Optional, alphas, alphanums
from scipy.io import arff
import collections
import csv
import math
import os
import pandas as pd

def main():
    malware()

### NSL-KDD
def nsl():
    nsldata_train = arff.loadarff('data/nsl/KDDTrain+.arff')
    nsldf_train = pd.DataFrame(nsldata_train[0]).infer_objects()

    nsldata_test = arff.loadarff('data/nsl/KDDTest+.arff')
    nsldf_test = pd.DataFrame(nsldata_test[0]).infer_objects()

    # Normalise & Standardise
    train_norm = pd.DataFrame()
    test_norm = pd.DataFrame()
    for k,v in nsldf_train.dtypes.items():
        if k == 'class':
            train_norm[k] = nsldf_train[k].astype('category')
            test_norm[k] = nsldf_test[k].astype('category')
        elif str(v) == 'float64':
            mean = nsldf_train[k].mean()
            std = nsldf_train[k].std()
            if std != 0:
                train_norm[k] = (nsldf_train[k] - mean)/std
                test_norm[k] = (nsldf_test[k] - mean)/std
        else:
            onehottrain = pd.get_dummies(nsldf_train[k])
            onehottest = pd.get_dummies(nsldf_test[k])

            onehottrain.columns = [k + '_' + c.decode('utf-8') for c in onehottrain.columns]
            onehottest.columns = [k + '_' + c.decode('utf-8') for c in onehottest.columns]

            for col in (onehottrain.columns.symmetric_difference(onehottest.columns)):
                onehottest[col] = 0
            train_norm = train_norm.join(onehottrain)
            test_norm = test_norm.join(onehottest)
    print(train_norm.columns)
    assert(train_norm.shape[1] == test_norm.shape[1]) # Make sure columns line up


### Microsoft Malware
# Classes
# 1 Ramnit
# 2 Lollipop
# 3 Kelihos_ver3
# 4 Vundo
# 5 Simda
# 6 Tracur
# 7 Kelihos_ver1
# 8 Obfuscator.ACY
# 9 Gatak

def aimf(entry):
    aimfd = dict([(i,0) for i in range(256)])
    with open(entry.path, encoding = 'ISO-8859-1') as f:
        source = f.readlines()
        hex_integer = Word(hexnums) + WordEnd() # use WordEnd to avoid parsing leading a-f of non-hex numbers as a hex
        line = '.text:' + hex_integer + Optional((hex_integer*(1,))('instructions') + Word(alphas,alphanums)('opcode'))
        for source_line in source:
            if source_line.startswith('.text:'):
                result = line.parseString(source_line)
                if 'opcode' in result:
                    aimfd[int(result.instructions.asList()[0], 16)] += 1

    d = dict(collections.Counter(aimfd))
    return d

def unigram(entry):
    aimf_res = aimf(entry)
    return {k:1 + math.log(v) if v != 0 else 0 for k, v in aimf_res.items()}

def malware():
    labels = pd.read_csv('data/mal/trainLabels.csv')
    files = os.scandir('data/mal/datasample')
    #files = os.scandir('data/mal/train')
    i = 0
    with open('data/mal/train.csv', 'w') as traincsv:
        trainwrite = csv.writer(traincsv)
        trainwrite.writerow([x for x in range(256)] + ['class'])
        for entry in files:
            print(i, entry.name)
            i += 1
            if entry.name.endswith('.asm'):
                typeclass = labels[labels['Id'] == os.path.splitext(entry.name)[0]].iloc[0]['Class']
                d = unigram(entry)|{'class':typeclass}
                trainwrite.writerow(d.values())

if __name__ == '__main__':
    main()
