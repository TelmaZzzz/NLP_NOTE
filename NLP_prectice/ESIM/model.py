import torch
import logging
import torch.nn as nn
import torch.nn.functional as F 

class ESIM(nn.Module):
    def __init__(self, args):
        super(ESIM, self).__init__()
        self._embed = nn.Embedding(args.embed_num, args.embed_dim)
        self._biLSTM1 = nn.LSTM(args.embed_dim, args.hidden_dim // 2, bidirectional=True, batch_first=True)
        self._biLSTM2 = nn.LSTM(args.hidden_dim * 4, args.hidden_dim // 2, bidirectional=True, batch_first=True)
        self._FC = nn.Linear(args.hidden_dim * 4, args.class_num)
    
    def _softmax_attention(self, xa, xb):
        # xa: batch_size * lena * dim
        # xb: batch_size * lenb * dim
        e = torch.matmul(xa, xb.transpose(1, 2)) # batch_size * lena * lenb
        weighta = e.softmax(-1) # batch_size * lena * lenb
        weightb = e.transpose(1, 2).softmax(-1) # batch_size * lenb * lena
        logging.debug("weighta:{}".format(weighta.size()))
        logging.debug("weightb:{}".format(weightb.size()))
        return torch.matmul(weighta, xb), torch.matmul(weightb, xa)

    def _pooling(self, v):
        return torch.cat([F.avg_pool1d(v.transpose(1, 2), v.size(1)).squeeze(-1), F.max_pool1d(v.transpose(1, 2), v.size(1)).squeeze(-1)], -1)
    
    def forward(self, xa, xb):
        xa, _ = self._biLSTM1(self._embed(xa))
        xb, _ = self._biLSTM1(self._embed(xb))
        logging.debug("xa in LSTM1:{}".format(xa.size()))
        logging.debug("xb in LSTM2:{}".format(xb.size()))

        # ya: batch_size * lena * dim
        # yb: batch_size * lenb * dim
        ya, yb = self._softmax_attention(xa, xb)
        logging.debug("ya:{}".format(ya.size()))
        va, _ = self._biLSTM2(F.relu(torch.cat([xa, ya, xa - ya, xa * ya], -1)))
        vb, _ = self._biLSTM2(F.relu(torch.cat([xb, yb, xb - yb, xb * yb], -1)))
        # va: batch_size * lena * dim
        # vb: batch_size * lenb * dim
        logging.debug("size va:{}".format(va.size()))

        v = torch.cat([self._pooling(va), self._pooling(vb)], -1)
        logging.debug("size v:{}".format(v.size()))
        terminal = self._FC(v)
        logging.debug("terminal:{}".format(terminal.size()))
        return terminal

        


