import os
import logging
import argparse
import pandas as pd
import numpy as np 
import torch
import torch.nn as nn 

parser = argparse.ArgumentParser()
parser.add_argument('--train-load-path', type=str)
parser.add_argument('--test-load-path', type=str)
parser.add_argument('--batch-size', type=int)

args = parser.parse_args()
def load_data(path):
    data = pd.read_csv(path, sep=',')
    data = data.reindex(
        np.random.permutation(data.index)
    )
    train_data = data.head(data.index*0.9)
    validation_data = data.tail(data.index*0.1)
    train_features = train_data['text']
    train_targets = train_data['star']
    validation_features = validation_data['text']
    validation_targets = validation_data['star']
    return train_features, train_targets, validation_features, validation_targets

def my_input_fn():
    pass

if __name__ == '__main__':
    train_features, train_targets, validation_feature, validation_targets = \
        load_data(args.train_load_path)




    
