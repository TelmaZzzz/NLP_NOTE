import os
import argparse
import logging
import torch
import warnings
import model
import train
from torchtext import data

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--train-data", type=str)
parser.add_argument("--valid-data", type=str)
parser.add_argument("--test-data", type=str)
parser.add_argument("--embed-dim", type=int)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--type", type=str, default="train")
parser.add_argument("--learning-rate", type=float, default=0.01)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--epoch", type=int, default=300)
parser.add_argument("--hidden-dim", type=int, default=100)
parser.add_argument("--eval-steps", type=int, default=500)

class MyDataset(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, url, text_field, target_field, examples=None, **kwargs):
        def readfromfile(url):
            with open(url, "r", encoding="utf-8") as f:
                return f.readlines()
        
        text_field.tokenize = lambda x:x.strip().split()
        fields = [("text", text_field), ("target", target_field)]
        if examples is None:
            texts = readfromfile(url[0])
            targets = readfromfile(url[1])
            assert len(texts) == len(targets)
            examples = [data.Example.fromlist(list(item), fields) for item in zip(texts, targets)]
        super(MyDataset, self).__init__(examples, fields, **kwargs)


if __name__ == "__main__":
    args = parser.parse_args()
    train_url = ("/".join([args.train_data, "seq.in"]), "/".join([args.train_data, "seq.out"]))
    valid_url = ("/".join([args.valid_data, "seq.in"]), "/".join([args.valid_data, "seq.out"]))
    text_field = data.Field(lower=True)
    target_field = data.Field(unk_token=None, pad_token=None)
    logging.info("Load Data ...")
    train_dataset = MyDataset(train_url, text_field, target_field)
    valid_dataset = MyDataset(valid_url, text_field, target_field)
    logging.info("Success To Load Data")
    text_field.build_vocab(train_dataset, valid_dataset)
    target_field.build_vocab(train_dataset, valid_dataset, specials=['<START>', '<STOP>'])
    train_iter, valid_iter = data.Iterator.splits((train_dataset, valid_dataset),
                                                  batch_sizes=(args.batch_size, args.batch_size),
                                                  repeat=False)
    args.embed_num = len(text_field.vocab)
    args.class_num = len(target_field.vocab)
    args.tag_stoi = target_field.vocab.stoi
    BiLSMT_CRF = model.BiLSTM_CRF(args)
    if args.type == 'train':
        train.train(train_iter, valid_iter, BiLSMT_CRF, args)
        logging.info("Successful Training")

