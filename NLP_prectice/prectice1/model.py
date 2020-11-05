import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self._args = args
        V = args.embeding_num
        D = args.embeding_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self._embed = nn.Embedding(V, D)
        self._convs = nn.ModuleList([nn.Conv2d(Ci, Co, (Ki, D)) for Ki in Ks])
        self._dropout = nn.Dropout(args.dropout)
        self._linear = nn.Linear(len(Ks)*Co, C)

    def forward(self, x):
        x = self._embed(x) # (N, V, D)
        x = x.unsqueeze(1) # (N, Ci, V, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self._convs] # (N, Co, V)*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # (N, Co)*len(Ks)
        x = torch.cat(x, 1) # (N, Co*len(Ks))
        # x = self._dropout(x)
        logit = self._linear(x) # (N, C)
        return logit

