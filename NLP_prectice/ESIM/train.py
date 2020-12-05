import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval(valid_iter, model, args):
    logging.info("Start evalution ...")
    correct = 0
    total = 0
    for iter in valid_iter:
        seq1, seq2, target = iter.seq1, iter.seq2, iter.target
        seq1.t_()
        seq2.t_()
        target = target.squeeze(0)
        logit = model(seq1, seq2)
        correct += (torch.max(logit, 1)[1].view(target.size()) == target).sum()
        total += target.size()[0]
    logging.info("auc:{:.4f}%({}/{})".format(correct / total * 100, correct, total))
    logging.info("Success evalution ...")

def train(train_iter, valid_iter, model, args):
    logging.info("Start training ...")
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    steps = 0
    average_loss = 0
    for epoch in range(args.epoch):
        for iter in train_iter:
            seq1, seq2, target = iter.seq1, iter.seq2, iter.target
            seq1.t_()
            seq2.t_()
            target.t_()
            logit = model(seq1, seq2)
            logging.debug("target:{}".format(target.size()))
            logging.debug("logit:{}".format(logit.size()))
            loss = F.cross_entropy(logit, target.squeeze(-1))
            average_loss += loss
            loss.backward()
            steps += 1
            if steps % args.eval_step == 0:
                eval(valid_iter, model, args)

            if steps % args.log_step == 0:
                logging.info("loss:{:.4f}".format(average_loss / args.log_step))
                average_loss = 0
    logging.info("Success training ...")
            

