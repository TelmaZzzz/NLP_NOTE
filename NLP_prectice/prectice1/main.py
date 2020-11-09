import os
import logging
import argparse
import random
import pandas as pd
import numpy as np 
import re
import torch
import torch.nn as nn
import warnings
from torchtext import data
import train
from model import CNN

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument('--train-load-path', type=str)
parser.add_argument('--test-load-path', type=str)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--split-pro', type=float, default=0.9)
parser.add_argument('--embeding-dim', type=int, default=133)
parser.add_argument('--kernel-num', type=int, default=100)
parser.add_argument('--kernel-sizes', type=str, default="3,4,5")
parser.add_argument('--fix-length', type=int, default=500)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--type', type=str, default='train')
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=256)
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--log-interval', type=str, default=20)
parser.add_argument('--eval-interval', type=int, default=100)

class MyDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, examples, load_path, text_field, target_field, **kwargs):
        def clean_text(text):
            text = re.sub(r"[^A-Za-z0-9<>,'\(\)\?\.\! ]*", "", text)
            text = re.sub(r"\.", " . ", text)
            text = re.sub(r"\?", " ? ", text)
            text = re.sub(r"<", " < ", text)
            text = re.sub(r">", " > ", text)
            text = re.sub(r"\!", " ! ", text)
            text = re.sub(r"'", " ' ", text)
            text = re.sub(r",", " , ", text)
            text = re.sub(r"\(", " ( ", text)
            text = re.sub(r"\)", " ) ", text)
            return text.strip().split()
        
        text_field.tokenize = lambda x:clean_text(x)
        fields = [('text', text_field), ('target', target_field)]
        if len(examples) == 0:
            dataframe = pd.read_csv(load_path)
            text_dataframe = dataframe['text']
            star_dataframe = dataframe['star']
            examples = [data.Example.fromlist([text, star_dataframe[inx]], fields) for inx, text in enumerate(text_dataframe)]
        super(MyDataset, self).__init__(examples, fields, **kwargs)
    
    @classmethod
    def split(cls, load_path, text_field, target_field, split_pro=0.1, shuffle=True, epoch=None, **kwargs):
        examples = []
        examples = cls(examples, load_path, text_field, target_field, **kwargs).examples
        if shuffle:
            random.shuffle(examples)
        dev_index = int((1.0-split_pro)*len(examples))
        return cls(examples[:dev_index], load_path, text_field, target_field, **kwargs), \
            cls(examples[dev_index:], load_path, text_field, target_field, **kwargs)


if __name__ == '__main__':
    args=parser.parse_args()
    text_field = data.Field(sequential=True, fix_length=args.fix_length, lower=True)
    target_field = data.Field(sequential=False)
    train_data, validation_data = MyDataset.split(args.train_load_path, text_field, target_field)
    text_field.build_vocab(train_data, validation_data)
    target_field.build_vocab(train_data, validation_data)
    train_iter, validation_iter = data.Iterator.splits((train_data, validation_data),
                                                        batch_sizes=(args.batch_size, args.batch_size),
                                                        repeat=False)
    args.embeding_num = len(text_field.vocab)
    args.class_num = len(target_field.vocab)
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    logging.info('embeding_num is :{}\nclass_num is :{}'.format(args.embeding_num, args.class_num))
    cnn = CNN(args)
    if(args.type == 'train'):
        train.train(train_iter, validation_iter, cnn, args)
        logging.info('over')

    elif(args.type == 'predict'):
        pass


    
