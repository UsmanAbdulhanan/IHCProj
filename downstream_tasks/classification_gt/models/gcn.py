import torch
import torch.nn as nn
import torch.nn.functional as F

# GCN basic operation
class GCNBlock(nn.Module):
    def __init__(
            self, input_dim, output_dim,
            batch_norm=False, add_self=False, normalize_embedding=False,
            relu=False, bias=True, dropout=0.0,
            **kwargs
    ):
        super(GCNBlock,self).__init__()

        self.add_self = add_self
        self.dropout = dropout
        self.relu = relu
        self.bn = batch_norm
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        if self.bn:
            self.bn_layer = torch.nn.BatchNorm1d(output_dim)

        self.normalize_embedding = normalize_embedding
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj, mask):
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)

        if self.bn:
            index = mask.sum(dim=1).long().tolist()
            bn_tensor_bf = mask.new_zeros((sum(index),y.shape[2]))
            bn_tensor_af = mask.new_zeros(*y.shape)
            start_index = []
            ssum = 0
            for i in range(x.shape[0]):
                start_index.append(ssum)
                ssum += index[i]
            start_index.append(ssum)

            for i in range(x.shape[0]):
                bn_tensor_bf[start_index[i]:start_index[i+1]] = y[i, 0:index[i]]
            bn_tensor_bf = self.bn_layer(bn_tensor_bf)

            for i in range(x.shape[0]):
                bn_tensor_af[i, 0:index[i]] = bn_tensor_bf[start_index[i]:start_index[i+1]]
            y = bn_tensor_af

        if self.dropout > 0.001:
            y = self.dropout_layer(y)
        if self.relu:
            y = torch.nn.functional.relu(y)
        return y
