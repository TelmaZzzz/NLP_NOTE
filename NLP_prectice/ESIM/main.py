import logging
import torch
import warnings
import argparse
import jsonlines
from torchtext import data
from model import ESIM
import train

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--train-url", type=str)
parser.add_argument("--valid-url", type=str)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--embed-dim", type=int, default=200)
parser.add_argument("--hidden-dim", type=int, default=200)
parser.add_argument("--learning-rate", type=int, default=0.01)
parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--log-step", type=int, default=10)
parser.add_argument("--eval-step", type=int, default=20)

class MyDataset(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.seq1)

    def __init__(self, url, seq_field, target_field, **kwargs):
        seq_field.tokenize = lambda x:x.strip().split()
        fields = [("seq1", seq_field), ("seq2", seq_field), ("target", target_field)]
        seq1_list = []
        seq2_list = []
        target_list = []
        with open(url, "r", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                seq1_list.append(item["sentence1"])
                seq2_list.append(item["sentence2"])
                target_list.append(item["gold_label"])
        examples = [data.Example.fromlist(list(item), fields) for item in zip(seq1_list, seq2_list, target_list)]
        super(MyDataset, self).__init__(examples, fields, **kwargs)
        
        
if __name__ == "__main__":
    args = parser.parse_args()
    seq_field = data.Field(lower=True)
    target_field = data.Field(unk_token=None, pad_token=None)
    logging.info("Start prepare dataset")
    train_dataset = MyDataset(args.train_url, seq_field, target_field)
    valid_dataset = MyDataset(args.valid_url, seq_field, target_field)
    seq_field.build_vocab(train_dataset, valid_dataset)
    target_field.build_vocab(train_dataset, valid_dataset)
    train_iter, valid_iter = data.Iterator.splits((train_dataset, valid_dataset),
                                                  batch_sizes=(args.batch_size, args.batch_size),
                                                  repeat=False)
    args.class_num = len(target_field.vocab)
    args.embed_num = len(seq_field.vocab)
    logging.debug(target_field.vocab.stoi)
    logging.info("Success")
    model = ESIM(args)
    train.train(train_iter, valid_iter, model, args)    
    
