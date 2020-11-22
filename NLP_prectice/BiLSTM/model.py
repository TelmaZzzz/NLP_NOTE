import torch
import logging
import torch.nn as nn
import torch.nn.functional as F 

START = "<START>"
STOP = "<STOP>"

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_expand = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_expand)))

class BiLSTM_CRF(nn.Module):
    def __init__(self, args):
        super(BiLSTM_CRF, self).__init__()
        self._hidden_dim = args.hidden_dim
        self._batch_size = args.batch_size
        self._embed = nn.Embedding(args.embed_num, args.embed_dim)
        self._lstm = nn.LSTM(args.embed_dim, self._hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self._hid2tag = nn.Linear(self._hidden_dim, args.class_num)
        self._tag_stoi = args.tag_stoi
        self._tag_size = len(self._tag_stoi)
        self._transitions = nn.Parameter(torch.randn(self._tag_size, self._tag_size))
        self._transitions.data[self._tag_stoi[START], :] = -10000.
        self._transitions.data[:, self._tag_stoi[STOP]] = -10000.
    
    def _get_lstm(self, seq):
        self._hidden = (torch.randn(2, 1, self._hidden_dim // 2), \
                    torch.randn(2, 1, self._hidden_dim // 2))
        embed = self._embed(seq).view(len(seq), 1, -1) # 1*len*dim
        # embed = torch.unsqueeze(embed, 1)
        logging.debug("embed size {}".format(embed.size()))
        logging.debug("hidden size {}".format(self._hidden[1].size()))
        lstm_out, self._hidden = self._lstm(embed, self._hidden)
        lstm_out = lstm_out.view(len(seq), self._hidden_dim)
        lstm_out = self._hid2tag(lstm_out)
        return lstm_out

    def neg_log_likelihood(self, seq, tag):
        lstm_out = self._get_lstm(seq)
        forwad_score = self._forward_alg(lstm_out)
        now_score = self._seq_score(lstm_out, tag)
        return forwad_score - now_score

    def _forward_alg(self, seq):
        forward_var = torch.full((1, self._tag_size), -10000.)
        forward_var[0][self._tag_stoi[START]] = 0.
        for feat in seq:
            now_forward_list = []
            for next_tag in range(self._tag_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self._tag_size)
                trans_score = self._transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                now_forward_list.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(now_forward_list).view(1, -1)
        terminal_var = forward_var + self._transitions[self._tag_stoi[STOP]]
        return log_sum_exp(terminal_var).view(1)
    
    def _seq_score(self, seq, tags):
        score = torch.zeros(1)
        logging.debug("START size:{}".format(torch.tensor([self._tag_stoi[START]], dtype=torch.long).size()))
        logging.debug("tags size:{}".format(tags.size()))
        tags = torch.cat([torch.tensor([self._tag_stoi[START]], dtype=torch.long), tags])
        logging.debug(seq.size())
        for i, feat in enumerate(seq):
            score = score + self._transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
        score = score + self._transitions[self._tag_stoi[STOP], tags[-1]]
        return score

    def _viterbi_decode(self, seq):
        best_path_arr = []
        forward_var = torch.full((1, self._tag_size), -10000.)
        forward_var[0][self._tag_stoi[START]] = 0.
        for feat in seq:
            best_path_feat = []
            next_tag_list = []
            for next_tag in range(self._tag_size):
                next_tag_var = forward_var + self._transitions[next_tag]
                best_path_idx = argmax(next_tag_var)
                best_path_feat.append(best_path_idx)
                logging.debug(next_tag_var.size())
                next_tag_list.append(next_tag_var[0][best_path_idx].view(1))
            best_path_arr.append(best_path_feat)
            forward_var = (torch.cat(next_tag_list)+feat).view(1, -1)
        
        terminal_var = forward_var + self._transitions[self._tag_stoi[STOP]]
        best_idx = argmax(terminal_var)
        score = terminal_var[0][best_idx]
        best_path = [best_idx]
        for iter in reversed(best_path_arr):
            best_idx = iter[best_idx]
            best_path.append(best_idx)
        best_path.pop()
        best_path.reverse()
        return score, best_path

    def forward(self, seq):
        lstm_out = self._get_lstm(seq)
        return self._viterbi_decode(lstm_out)
