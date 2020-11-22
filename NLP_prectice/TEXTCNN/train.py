import os
import logging
import argparse
import pandas as pd
import numpy as np 
import torch
import sys
import torch.nn as nn 
import torch.nn.functional as F 

def train(train_iter, validation_iter, model, args):
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    logging.info('training...')
    step = 0
    auc_maxn = 0
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            model.train()
            feature, target = batch.text, batch.target
            feature.t_()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            step += 1
            if step % args.log_interval == 0:
                logging.info('log calculate...')
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                auc = 100.0 * corrects/batch.batch_size
                logging.info(
                    'step:{}. loss:{}. auc:{}({}/{})'.format(step, loss.item(), auc.item(), 
                    corrects.item(), batch.batch_size)
                )
            
            if step % args.eval_interval == 0:
                auc = eval(validation_iter, model, args)
                if auc > auc_maxn:
                    auc_maxn = auc
                    save(model, args, 'best_auc')
    save(model, args, 'last')

def eval(validation_iter, model, args):
    model.eval()
    avg_loss, auc, size = 0, 0, 0
    logging.info('evalution...')
    for batch in validation_iter:
        feature, target = batch.text, batch.target
        feature.t_()

        logit = model(feature)
        loss = F.cross_entropy(logit, target)
        avg_loss += loss
        auc += (torch.max(logit, 1)[1].view(target.size()).data == \
            target.data).sum().item()
        size += target.size()[0]
        # print(auc, size)
    logging.info('success')
    avg_loss /= len(validation_iter)
    auc = auc * 100 / size
    logging.info('avg_loss:{}. auc:{}'.format(avg_loss.item(), auc))
    return auc

def save(model, args, name):
    if args.save_path:
        logging.info('save model...')
        if not os.path.isdir(args.save_path):
            os.mkdir(args.save_path)
        save_path = '/'.join([args.save_path, name])
        torch.save(model.state_dict(), save_path)



    
