import torch
import logging
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

def eval(valid_iter, model, args):
    logging.info("Start eval ...")
    correct_num = 0
    total_num = 0
    total_score = 0
    for iter in valid_iter:
        sentence, target = iter.text, iter.target
        target = torch.squeeze(target, 1)
        logging.debug("target data:{}".format(target.item))
        with torch.no_grad():
            score, path = model(sentence)
            correct_num += (target == torch.tensor(path)).sum()
            total_num += len(path)
            total_score += score
    acc = correct_num / total_num * 100
    logging.info("acc: {}%({}/{}). average_score:{}".format(
        acc, correct_num, total_num, total_score/len(valid_iter)
    ))
    logging.info("End eval ...")

            


def train(train_iter, valid_iter, model, args):
    logging.info("Start training ...")
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps = 0
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
            logging.info("loss: {}".format(loss.item()))
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.eval_steps == 0:
                eval(valid_iter, model, args)
            


            # with torch.no_grad():
            #     loss, path = model(sentence)
            #     logging.info("loss is:{}\nbest path is:{}\ntarget is:{}".format(
            #         loss, list(path), target
            #     ))