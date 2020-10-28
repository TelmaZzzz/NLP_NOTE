import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load-path', type=str)
parser.add_argument('--file-name', type=str)
args = parser.parse_args()
NEG = 'neg'
POS = 'pos'
UNSUP = 'unsup'
columns = ['text', 'star', 'labels']
datas = []

def gen(label):
    path_label = '/'.join([args.load_path, label])
    path_labels = os.listdir(path_label)
    for path in path_labels:
        real_path = '/'.join([path_label, path])
        with open(real_path, "r", encoding='UTF-8') as f:
            feature = f.read()
            _, value, _ = path.replace('.', '_').split('_')
            data = []
            data.append(feature)
            data.append(value)
            data.append(label)
            datas.append(data)

if __name__ == "__main__":
    path_gen = '/'.join([args.load_path, args.file_name])
    path_gen = '.'.join([path_gen, "csv"])
    gen(NEG)
    gen(POS)
    gen(UNSUP)
    data_frame = pd.DataFrame(columns=columns, data=datas)
    data_frame.to_csv(path_gen, sep=",")




