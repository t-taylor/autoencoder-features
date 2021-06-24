### Part 1 initial dataset analysis
# * Importing datasets
# * Feature analysis
# * Normalisation and Standardization
# * Training & testing without autoencoder (?)

from pyparsing import Word, hexnums, WordEnd, Optional, alphas, alphanums
from scipy.io import arff
import collections
import math
import os
import pandas as pd

def main():
    malware()

### NSL-KDD
def nsl():
    nsldata = arff.loadarff('data/nsl/KDDTrain+.arff')
    nsldf = pd.DataFrame(nsldata[0]).infer_objects()
    print(nsldf.dtypes)

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
    aimf = dict([(i,0) for i in range(256)])
    with open(entry.path, encoding = 'ISO-8859-1') as f:
        source = f.readlines()
        hex_integer = Word(hexnums) + WordEnd() # use WordEnd to avoid parsing leading a-f of non-hex numbers as a hex
        line = '.text:' + hex_integer + Optional((hex_integer*(1,))('instructions') + Word(alphas,alphanums)('opcode'))
        for source_line in source:
            if 'text' not in source_line:
                break;
            result = line.parseString(source_line)
            if 'opcode' in result:
                aimf[int(result.instructions.asList()[0], 16)] += 1

    d = dict(collections.Counter(aimf))
    return d

def unigram(entry):
    aimf_res = aimf(entry)
    return {k:1 + math.log(v) if v != 0 else 0 for k, v in aimf_res.items()}

def malware():
    labels = pd.read_csv('data/mal/trainLabels.csv')
    XY = pd.DataFrame()
    #for entry in os.scandir('data/mal/train'):
    for entry in os.scandir('data/mal/datasample'):
        if entry.name.endswith('.asm'):
            typeclass = labels[labels['Id'] == os.path.splitext(entry.name)[0]].iloc[0]['Class']
            XY = XY.append({'class':typeclass}|unigram(entry), ignore_index=True)

    print(XY)

if __name__ == '__main__':
    main()
