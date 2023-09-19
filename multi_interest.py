import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import pickle
import gzip
import numpy as np
import time

from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv
import dgl.nn as dglnn
import dgl.function as fn
import dgl
from utils import *

def G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence sparse matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    DV = torch.sum(H, dim=1, keepdim=True) + 1e-5
    DE = torch.sum(H, dim=0, keepdim=True)+ 1e-5

    invDE = torch.diag(DE.pow(-1).reshape(-1))
    DV2 = torch.diag(DV.pow(-1).reshape(-1))
    HT = H.transpose(0,1)

    G = DV2[:1,:].matmul(H).matmul(invDE).matmul(HT).matmul(DV2)

    return G
class SelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.,anchor=False):
        super().__init__()
        self.anchor=anchor
        if anchor:
            self.scorer = nn.Linear(d_hid*2, 1)
        else:
            self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters(d_hid)
    def reset_parameters(self,d_hid):
        if self.anchor:
            stdv = 1. / math.sqrt(d_hid*2)
        else:
            stdv = 1. / math.sqrt(d_hid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_seq,anchor=None,s=None):
        batch_size, seq_len, feature_dim = input_seq.size()
        input_seq = self.dropout(input_seq)
        
        if anchor!=None:
            size=input_seq.shape[1]
            anchor=anchor.repeat(1,size,1)
            seq=torch.cat((input_seq,anchor),2)
            # enablePrint()
            # ipdb.set_trace()
            scores = self.scorer(seq.contiguous().view(-1, feature_dim*2)).view(batch_size, seq_len)+s
        else:
            scores = self.scorer(input_seq.contiguous().view(-1, feature_dim)).view(batch_size, seq_len)
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(input_seq).mul(input_seq).sum(1)
        return context,scores # 既然命名为context就应该是整句的表示



class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class HGNN_conv(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(HGNN_conv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #hyperG=G_from_H(adj)
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def outprop(self, H_input, adj):
        #hyperG=G_from_H(adj)
        output = torch.sparse.mm(adj, H_input)
        return output

class GraphEncoder(Module):
    def __init__(self,graph,device, entity, emb_size, kg, embeddings=None, fix_emb=True, seq='rnn', gcn=True, hidden_size=100, layers=1, rnn_layer=1,u=None,v=None,f=None):
        super(GraphEncoder, self).__init__()

        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        # self.eps = 0.0
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        rel_names=['interact','friends','like','belong_to']
        self.G=graph.to(device)
        self.conv1 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(emb_size, hidden_size) for rel in rel_names},
                                           aggregate='mean')

        self.embedding = nn.Embedding(entity, emb_size, padding_idx=entity-1)
        if embeddings is not None:
            print("pre-trained embeddings")
            self.embedding.from_pretrained(embeddings,freeze=fix_emb)
        self.layers = layers
        self.user_num = u
        self.item_num = v
        self.PADDING_ID = entity-1
        self.device = device
        self.seq = seq
        self.gcn = gcn
        self.hidden_size=hidden_size

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        #self.fc_frd = nn.Linear(hidden_size, hidden_size)
        if self.seq == 'rnn':
            self.rnn = nn.GRU(hidden_size, hidden_size, rnn_layer, batch_first=True)
        elif self.seq == 'transformer':
            self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=400), num_layers=rnn_layer)

        if self.gcn:
            indim, outdim = emb_size, hidden_size
            self.gnns = nn.ModuleList()
            self.hypergnns=nn.ModuleList()
            for l in range(layers):
                self.hypergnns.append(HGNN_conv(indim, outdim))
                indim = outdim
        else:
            self.fc2 = nn.Linear(emb_size, hidden_size)
        #self.num_pers=4
        #self.multi_head_self_attention_init = nn.ModuleList([SelfAttention(self.hidden_size, 0.3) for _ in range(self.num_pers)])
        #self.multi_head_self_attention = nn.ModuleList([SelfAttention(self.hidden_size, 0.3,anchor=True) for _ in range(self.num_pers)])

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def ssl(self, b_state):
        tau = 0.6  # default = 0.8
        f = lambda x: torch.exp(x / tau)
        hyper_batch_output = []
        for s in b_state:
            # neighbors, adj = self.get_state_graph(s)
            hyperneigh, HT = s['hyperneigh'].to(self.device), s['hyperHT'].to(self.device)
            hyper_input_state = self.embedding(hyperneigh)
            if self.gcn:
                for hypergnn in self.hypergnns:
                    hyper_output_state = hypergnn(hyper_input_state, HT)
                    # cand_act_embedding=hypergnn.outprop(hyper_output_state, G)
                    hyper_input_state = hyper_output_state
                hyper_batch_output.append(hyper_output_state)


        ssl_loss_set = []
        for s, o in zip(b_state, hyper_batch_output):
            # seq_embeddings.append(o[:len(s['cur_node']),:][None,:])
            ssl_loss = torch.tensor(0.0).cuda()
            cnt = 1e-10
            if len(s['acc_feature'])>0 and len(s['rej_feature'])>0:
                feature_pos_emb=o[:len(s['acc_feature'])]
                feature_posctr_emb=feature_pos_emb.mean(axis=0, keepdim=True)
                feature_neg_emb=o[len(s['acc_feature']):len(s['rej_feature'])]
                feature_negctr_emb = feature_neg_emb.mean(axis=0, keepdim=True)
                pos_pos_sim=f(self.sim(feature_posctr_emb, feature_pos_emb))
                pos_neg_sim=f(self.sim(feature_posctr_emb, feature_neg_emb))
                neg_pos_sim = f(self.sim(feature_negctr_emb, feature_pos_emb))
                neg_neg_sim = f(self.sim(feature_negctr_emb, feature_neg_emb))
                ssl_loss+=-torch.log(pos_pos_sim.sum()/pos_neg_sim.sum())
                ssl_loss += -torch.log(neg_neg_sim.sum() / neg_pos_sim.sum())
                if len(s['friend'])>0:
                    feature_frd_emb=o[len(s['acc_feature'])+len(s['rej_feature']):len(s['acc_feature'])+len(s['rej_feature'])+len(s['friend'])]
                    feature_frdctr_emb=feature_frd_emb.mean(axis=0, keepdim=True)
                    frd_pos_sim=f(self.sim(feature_posctr_emb, feature_frd_emb))
                    frd_neg_sim=f(self.sim(feature_negctr_emb, feature_frd_emb))
                    ssl_loss += -torch.log(frd_pos_sim.sum() / frd_neg_sim.sum())
            ssl_loss_set.append(ssl_loss / cnt)
        ssl_loss_mean = torch.mean(torch.stack(ssl_loss_set))
        return ssl_loss_mean

    def forward(self, b_state, b_act=None):
        hyper_batch_output = []
        #hyper_cand_emb=[]
        for s in b_state:
            hyperneigh, HT = s['hyperneigh'].to(self.device), s['hyperHT'].to(self.device)
            hyper_input_state = self.embedding(hyperneigh)
            if self.gcn:
                for hypergnn in self.hypergnns:
                    hyper_output_state = hypergnn(hyper_input_state, HT)
                    #cand_act_embedding=hypergnn.outprop(hyper_output_state, G)
                    hyper_input_state = hyper_output_state
                hyper_batch_output.append(hyper_output_state)
                #hyper_cand_emb.append(cand_act_embedding)

        hyper_seq_embeddings = []
        neg_seq_embeddings = []
        friend_seq_embeddings=[]
        for s, o in zip(b_state, hyper_batch_output):
            hyper_seq_embeddings.append(o[:len(s['acc_feature']), :][None, :])
            if o[len(s['acc_feature']):len(s['acc_feature'])+len(s['rej_feature']), :].shape[0]>0:
                neg_seq_embeddings.append(o[len(s['acc_feature']):len(s['acc_feature'])+len(s['rej_feature']), :][None, :])
            else:
                neg_seq_embeddings.append(torch.zeros([1, 1, self.hidden_size]).to(self.device))
            if o[len(s['acc_feature'])+len(s['rej_feature']):len(s['acc_feature'])+len(s['rej_feature'])+len(s['friend']), :].shape[0]>0:
                friend_seq_embeddings.append(o[len(s['acc_feature'])+len(s['rej_feature']):len(s['acc_feature'])+len(s['rej_feature'])+len(s['friend']), :][None, :])
            else:
                friend_seq_embeddings.append(torch.zeros([1, 1, self.hidden_size]).to(self.device))
        if len(hyper_batch_output) > 1:
            hyper_seq_embeddings = self.padding_seq(hyper_seq_embeddings)
            neg_seq_embeddings = self.padding_seq(neg_seq_embeddings)
            friend_seq_embeddings = self.padding_seq(friend_seq_embeddings)
        hyper_seq_embeddings = torch.cat(hyper_seq_embeddings, dim=0)
        hyper_seq_embeddings = torch.mean(self.transformer(hyper_seq_embeddings), dim=1, keepdim=True)

        neg_seq_embeddings = torch.cat(neg_seq_embeddings, dim=0)
        neg_seq_embeddings = torch.mean(self.transformer(neg_seq_embeddings), dim=1, keepdim=True)

        friend_embeddings = torch.cat(friend_seq_embeddings, dim=0)
        friend_embeddings = torch.mean(self.transformer(friend_embeddings), dim=1, keepdim=True)

        # seq_embeddings = F.relu(self.fc1(seq_embeddings))
        hyper_seq_embeddings = F.relu(self.fc1(hyper_seq_embeddings)-self.fc1(neg_seq_embeddings)+self.fc1(friend_embeddings))
        return hyper_seq_embeddings

    def padding_seq(self, seq):
        padding_size = max([len(x[0]) for x in seq])
        padded_seq = []
        for s in seq:
            cur_size = len(s[0])
            emb_size = len(s[0][0])
            new_s = torch.zeros((padding_size, emb_size)).to(self.device)
            new_s[:cur_size,:] = s[0]
            padded_seq.append(new_s[None,:])
        return padded_seq

    def padding(self, cand_embs):
        pad_size = max([len(c) for c in cand_embs])
        padded_cand = []
        for c in cand_embs:
            cur_size = len(c)
            new_c = torch.zeros((pad_size-cur_size, c.size(1))).to(self.device)
            padded_cand.append(torch.cat((c,new_c),dim=0))
        return padded_cand
