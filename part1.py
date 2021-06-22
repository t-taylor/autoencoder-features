### Part 1 initial dataset analysis
# * Importing datasets
# * Feature analysis
# * Normalisation and Standardization
# * Training & testing without autoencoder (?)

from pyparsing import Word, hexnums, WordEnd, Optional, alphas, alphanums
from scipy.io import arff
import pandas as pd

def main():
    malware()

### NSL-KDD
def nsl():
    nsldata = arff.loadarff('data/nsl/KDDTrain+.arff')
    nsldf = pd.DataFrame(nsldata[0]).infer_objects()
    print(nsldf.dtypes)

### Microsoft Malware
def malware():
    with open('data/mal/datasample/0A32eTdBKayjCWhZqDOQ.asm', encoding = 'ISO-8859-1') as f:
        source = f.readlines()
        hex_integer = Word(hexnums) + WordEnd() # use WordEnd to avoid parsing leading a-f of non-hex numbers as a hex
        line = '.text:' + hex_integer + Optional((hex_integer*(1,))('instructions') + Word(alphas,alphanums)('opcode'))
        for source_line in source:
            if 'text' not in source_line:
                break;
            result = line.parseString(source_line)
            if 'opcode' in result:
                print(result.opcode, result.instructions.asList())

if __name__ == '__main__':
    main()
