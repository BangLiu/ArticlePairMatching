# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution


class SE_GCN(nn.Module):
    def __init__(self, args, W2V, sent_max_len, vector_size, nfeat, nfeat_v, nfeat_g,
                 nhid_vfeat, nhid_siamese, dropout_vfeat, dropout_siamese, nhid_final):
        super(SE_GCN, self).__init__()

        # network configure
        self.args = args
        self.embeddings = torch.FloatTensor(list(W2V.values()))
        self.num_filter = 32    # number of conv1d filters, also is the output size of Siamese encoder
        self.window_size = 1     # conv1d kernel window size
        self.step_size = sent_max_len  # sentence length
        self.transformed_encoding_size = 2 * self.num_filter

        # embedding
        self.embedding = nn.Embedding(self.embeddings.size(0), self.embeddings.size(1))
        self.embedding.weight = nn.Parameter(self.embeddings)
        self.embedding.weight.requires_grad = False
        self.use_cig = (self.args.use_vfeatures or self.args.use_siamese)  # if not use cig, it is a feature + NN model

        # encoding
        if args.use_siamese:
            # text pair embedding component
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=self.num_filter,
                          kernel_size=(self.window_size, vector_size)),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size=(self.step_size - self.window_size + 1, 1)))

            gcn_input_dim = self.transformed_encoding_size
            if args.use_gcn:
                assert args.num_gcn_layers > 0
                self.gc_w2v = nn.ModuleList()
                if args.num_gcn_layers == 1:
                    self.gc_w2v.append(GraphConvolution(gcn_input_dim, nhid_final))
                else:
                    self.gc_w2v.append(GraphConvolution(gcn_input_dim, nhid_siamese))
                    num_layer = 1
                    while num_layer < args.num_gcn_layers - 1:
                        self.gc_w2v.append(GraphConvolution(nhid_siamese, nhid_siamese))
                        num_layer += 1
                    self.gc_w2v.append(GraphConvolution(nhid_siamese, nhid_final))
            self.dropout_w2v = dropout_siamese

        # GCN transform
        if args.use_vfeatures:
            gcn_input_dim = nfeat_v
            if args.use_gcn:
                assert args.num_gcn_layers > 0
                self.gc_vfeat = nn.ModuleList()
                if args.num_gcn_layers == 1:
                    self.gc_vfeat.append(GraphConvolution(gcn_input_dim, nhid_final))
                else:
                    self.gc_vfeat.append(GraphConvolution(gcn_input_dim, nhid_vfeat))
                    num_layer = 1
                    while num_layer < args.num_gcn_layers - 1:
                        self.gc_vfeat.append(GraphConvolution(nhid_vfeat, nhid_vfeat))
                        num_layer += 1
                    self.gc_vfeat.append(GraphConvolution(nhid_vfeat, nhid_final))
            self.dropout_vfeat = dropout_vfeat

        # regression
        # TODO: add other types of pooling layers according to args
        regressor_input_dim = 0
        if args.use_gcn:
            if args.use_siamese and not args.use_vfeatures:
                regressor_input_dim = nhid_final
            if not args.use_siamese and args.use_vfeatures:
                regressor_input_dim = nhid_final
            if args.use_siamese and args.use_vfeatures:
                regressor_input_dim = 2 * nhid_final
            if not args.use_siamese and not args.use_vfeatures:
                regressor_input_dim = 0
            if args.use_gfeatures:
                regressor_input_dim += nfeat_g
        else:
            if args.use_siamese and not args.use_vfeatures:
                regressor_input_dim = self.transformed_encoding_size
            if not args.use_siamese and args.use_vfeatures:
                regressor_input_dim = nfeat_v
            if args.use_siamese and args.use_vfeatures:
                regressor_input_dim = self.transformed_encoding_size + nfeat_v
            if not args.use_siamese and not args.use_vfeatures:
                regressor_input_dim = 0
            if args.use_gfeatures:
                regressor_input_dim += nfeat_g

        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, nhid_final),
            nn.ReLU(),
            nn.Linear(nhid_final, 1),
            nn.Sigmoid())

    def forward(self, x_l, x_r, v_feature, adj, g_feature, g_vertex):
        if self.use_cig:
            if self.args.use_siamese:
                # encode a pair of word vector sequences by Siamese network
                x_l = self.embedding(x_l)
                x_r = self.embedding(x_r)
                batch, seq, embed = x_l.size()
                x_l = x_l.contiguous().view(batch, 1, seq, embed)
                x_r = x_r.contiguous().view(batch, 1, seq, embed)
                x_l = x_l.detach()
                x_r = x_r.detach()
                x_l = self.encoder(x_l)
                x_r = self.encoder(x_r)
                x_l = x_l.view(-1, self.num_filter)
                x_r = x_r.view(-1, self.num_filter)
                x_mul = torch.mul(x_l, x_r)
                x_dif = torch.abs(torch.add(x_l, -x_r))
                x_siamese = torch.cat([x_mul, x_dif], 1)  # vertex match vectors.

            if self.args.use_gcn and self.args.use_siamese:
                for n_l in range(self.args.num_gcn_layers):
                    x_siamese = self.gc_w2v[n_l](x_siamese, adj)
                    if n_l < self.args.num_gcn_layers - 1:
                        x_siamese = F.relu(x_siamese)
                        x_siamese = F.dropout(x_siamese, self.dropout_w2v, training=self.training)

            x_vfeat = v_feature
            if self.args.use_gcn and self.args.use_vfeatures:
                for n_l in range(self.args.num_gcn_layers):
                    x_vfeat = self.gc_vfeat[n_l](x_vfeat, adj)
                    if n_l < self.args.num_gcn_layers - 1:
                        x_vfeat = F.relu(x_vfeat)
                        x_vfeat = F.dropout(x_vfeat, self.dropout_vfeat, training=self.training)

            if self.args.use_vfeatures and not self.args.use_siamese:
                x = x_vfeat
            if not self.args.use_vfeatures and self.args.use_siamese:
                x = x_siamese
            if self.args.use_vfeatures and self.args.use_siamese:
                x = torch.cat([x_siamese, x_vfeat], 1)

            # graph aggregation.
            # TODO: different pooling layers
            x = torch.mean(x, dim=0)
            if self.args.use_gfeatures:
                x = torch.cat([x, g_feature])
        else:
            x = g_feature

        out = self.regressor(x)

        return out
