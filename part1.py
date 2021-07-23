### Part 1 initial dataset analysis
# * Importing datasets
# * Feature analysis
# * Normalisation and Standardization

from pyparsing import Word, hexnums, WordEnd, Optional, alphas, alphanums
from scipy.io import arff
import collections
import csv
import math
import os
import pandas as pd

_txtheaders =['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
              'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
              'num_failed_logins', 'logged_in', 'num_compromised',
              'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
              'num_shells', 'num_access_files', 'num_outbound_cmds',
              'is_host_login', 'is_guest_login', 'count', 'srv_count',
              'serror_rate', 'srv_serror_rate', 'rerror_rate',
              'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
              'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
              'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
              'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
              'dst_host_serror_rate', 'dst_host_srv_serror_rate',
              'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class',
              'difficulty']

def main():
    nsl_explore()

### NSL-KDD
def nsl_binary():
    nsldata_train = arff.loadarff('data/nsl/KDDTrain+.arff')
    nsldf_train = pd.DataFrame(nsldata_train[0]).infer_objects()

    nsldata_test = arff.loadarff('data/nsl/KDDTest+.arff')
    nsldf_test = pd.DataFrame(nsldata_test[0]).infer_objects()

    # Normalise & Standardise
    train_norm = pd.DataFrame()
    test_norm = pd.DataFrame()
    for k,v in nsldf_train.dtypes.items():
        if k == 'class' or k == 'difficulty':
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
    assert(train_norm.shape[1] == test_norm.shape[1]) # Make sure columns line up
    return (train_norm, test_norm)

def nsl_multiclass():
    nsldf_train = pd.read_csv('data/nsl/KDDTrain+.txt', names=_txtheaders)

    nsldf_test = pd.read_csv('data/nsl/KDDTest+.txt', names=_txtheaders)

    # Normalise & Standardise
    train_norm = pd.DataFrame()
    test_norm = pd.DataFrame()
    for k,v in nsldf_train.dtypes.items():
        if k == 'class':
            train_norm[k] = nsldf_train[k].astype('category')
            test_norm[k] = nsldf_test[k].astype('category')
        elif k == 'difficulty':
            pass
        elif str(v) == 'object':
            onehottrain = pd.get_dummies(nsldf_train[k])
            onehottest = pd.get_dummies(nsldf_test[k])

            onehottrain.columns = [k + '_' + c for c in onehottrain.columns]
            onehottest.columns = [k + '_' + c for c in onehottest.columns]

            for col in (onehottrain.columns.symmetric_difference(onehottest.columns)):
                onehottest[col] = 0
            train_norm = train_norm.join(onehottrain)
            test_norm = test_norm.join(onehottest)
        else:
            std = nsldf_train[k].std()
            if std != 0:
                # min max scaling
                minn = min(nsldf_train[k].min(), nsldf_test[k].min())
                maxn = max(nsldf_train[k].max(), nsldf_test[k].max())
                train_norm[k] = (nsldf_train[k] - minn)/(maxn - minn)
                test_norm[k] = (nsldf_test[k] - minn)/(maxn - minn)
                # normal distribution scaling
                # train_norm[k] = (nsldf_train[k] - mean)/std
                # test_norm[k] = (nsldf_test[k] - mean)/std

    assert(train_norm.shape[1] == test_norm.shape[1]) # Make sure columns line up
    return (train_norm, test_norm)

def multi_to_bin(data):
    train, test = data
    train = train.copy()
    test = test.copy()

    train['class'] = train['class'].map(lambda x: 1 if x not in 'normal' else 0)
    test['class'] = test['class'].map(lambda x: 1 if x not in 'normal' else 0)
    return train, test

def nsl_explore():
    (train, test) = nsl_multiclass()
    print('types of attack', list(train['class'].astype('category').cat.categories))


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
        i = 0
        for source_line in source:
            i += 1
            print(i)
            if source_line.startswith('.text:'):
                #print(source_line)
                result = line.parseString(source_line)
                if 'opcode' in result:
                    if len(result.instructions.asList()[0]) == 2:
                        aimfd[int(result.instructions.asList()[0], 16)] += 1

    d = dict(collections.Counter(aimfd))
    return d

def unigram(entry):
    aimf_res = aimf(entry)
    return {k:1 + math.log(v) if v != 0 else 0 for k, v in aimf_res.items()}

def malware():
    labels = pd.read_csv('data/mal/testLabels.csv')
    files = os.scandir('data/mal/datasample')
    #files = os.scandir('data/mal/train')
    i = 0
    with open('data/mal/train.csv', 'w') as traincsv:
        trainwrite = csv.writer(traincsv)
        trainwrite.writerow([x for x in range(256)] + ['class'])
        for entry in files:
            print(i, i / 21737, entry.name)
            i += 1
            if entry.name.endswith('.asm'):
                typeclass = labels[labels['Id'] == os.path.splitext(entry.name)[0]].iloc[0]['Class']
                d = unigram(entry)|{'class':typeclass}
                trainwrite.writerow(d.values())

def malware_df():
    mal = pd.read_csv('data/mal/train.csv')
    out = pd.DataFrame()
    for k in mal.columns:
        if k == 'class':
            out[k] = mal[k].astype('category')
        else:
            std = mal[k].std()
            if std != 0:
                # min max scaling
                minn = min(mal[k].min(), mal[k].min())
                maxn = max(mal[k].max(), mal[k].max())
                out[k] = (mal[k] - minn)/(maxn - minn)
                out[k] = (mal[k] - minn)/(maxn - minn)
    return out

if __name__ == '__main__':
    main()

