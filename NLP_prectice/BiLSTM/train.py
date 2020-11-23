import torch
import logging
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
O = 2

def get_metrics(path, target):
    TP, FP, FN = 0, 0, 0
    for pi, ti in zip(path, target):
        # logging.info("pi:{}, ti:{}".format(pi, ti.item()))
        if pi == ti.item() and pi != O:
            TP += 1
        if pi != ti.item() and pi != O:
            FP += 1
        if pi != ti.item() and ti.item() != O:
            FN += 1
    # logging.info("TP:{} FP:{} FN:{}".format(TP, FP, FN))
    return TP, FP, FN

def cal_metrics(TP, FP, FN):
    logging.info("TP:{} FP:{} FN:{}".format(TP, FP, FN))
    try:
        precision = TP / (TP + FP)
    except:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except:
        recall = 0
    try:
        F1 = 2 * recall * precision / (recall + precision)
    except:
        F1 = 0
    return {"precision":precision, "recall":recall, "F1":F1}

def eval(valid_iter, model, args):
    logging.info("Start eval ...")
    correct_num = 0
    total_num = 0
    total_score = 0
    TP, FP, FN = 0, 0, 0
    for iter in valid_iter:
        sentence, target = iter.text, iter.target
        target = torch.squeeze(target, 1)
        logging.debug("target data:{}".format(target.item))
        with torch.no_grad():
            score, path = model(sentence)
            correct_num += (target == torch.tensor(path)).sum()
            total_num += len(path)
            total_score += score
            nTP, nFP, nFN = get_metrics(path, target)
            TP += nTP
            FP += nFP
            FN += nFN
    acc = correct_num / total_num * 100
    metrics = cal_metrics(TP, FP, FN)
    logging.info("F1:{}%.  precision:{}%.  recall:{}%".format(metrics["F1"]*100, metrics["precision"]*100, metrics["recall"]*100))
    logging.info("acc: {}%({}/{}). average_score:{}".format(
        acc, correct_num, total_num, total_score/len(valid_iter)
    ))
    logging.info("End eval ...")

            


def train(train_iter, valid_iter, model, args):
    logging.info("Start training ...")
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps = 0
    average_loss = 0
    for epoch in range(args.epoch):
        logging.info("training epoch is: %s", epoch)
        for iter in train_iter:
            sentence, target = iter.text, iter.target
            # sentence.t_()
            target = torch.squeeze(target, 1)
            # sentence = torch.squeeze(sentence, 0)
            logging.debug(sentence.size())
            logging.debug(target.size())
            model.train()
            model.zero_grad()
            loss = model.neg_log_likelihood(sentence, target)
            steps += 1
            average_loss += loss.item()
            if steps % args.log_steps == 0:
                logging.info("{} step average_loss: {}".format(args.log_steps,
                    average_loss / args.log_steps))
                average_loss = 0
            loss.backward()
            optimizer.step()
            if steps % args.eval_steps == 0:
                eval(valid_iter, model, args)
            


            # with torch.no_grad():
            #     loss, path = model(sentence)
            #     logging.info("loss is:{}\nbest path is:{}\ntarget is:{}".format(
            #         loss, list(path), target
            #     ))