import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

word2vec = None


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        self.embedding = nn.Embedding(len(word2vec.vectors), embed)
        self.embedding.from_pretrained(torch.tensor(word2vec.vectors))
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])  # 偶数sin
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])  # 奇数cos
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        out = x + nn.Parameter(self.pe, requires_grad=False).to(torch.device("cuda:0"))
        out = self.dropout(out)
        return out


class Multi_Head_Attention(nn.Module):

    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0  # head数必须能够整除隐层大小
        self.dim_head = dim_model // self.num_head  # 按照head数量进行张量均分
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)  # Q，通过Linear实现张量之间的乘法，等同手动定义参数W与之相乘
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)  # 自带的LayerNorm方法

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1,
                   self.dim_head)  # reshape to batch*head*sequence_length*(embedding_dim//head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5  # 根号dk分之一，对应Scaled操作
        context = self.attention(Q, K, V, scale)  # Scaled_Dot_Product_Attention计算
        context = context.view(batch_size, -1, self.dim_head * self.num_head)  # reshape 回原来的形状
        out = self.fc(context)  # 全连接
        out = self.dropout(out)
        out = out + x  # 残差连接,ADD
        out = self.layer_norm(out)  # 对应Norm
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product'''

    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)  # 两层全连接
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Transformer(nn.Module):
    def __init__(self, model):
        global word2vec
        word2vec = model
        super(Transformer, self).__init__()
        self.postion_embedding = Positional_Encoding(100, 64, 0.5)
        self.encoder = Encoder(100, 5, 1024, 0.5)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(2)])  # 多次Encoder

        self.fc1 = nn.Linear(64 * 100, 3)

    def forward(self, x):
        out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)  # 将三维张量reshape成二维，然后直接通过全连接层将高维数据映射为classes
        out = self.fc1(out)
        return out
