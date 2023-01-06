import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed, embedding_dim=100, hidden_size=128):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.from_pretrained(torch.tensor(embed))
        self.lstm = nn.LSTM(embedding_dim, hidden_size, 2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        self.tanh2 = nn.Tanh()
        self.fc = nn.Linear(hidden_size * 2, 3)
        nn.init.uniform_(self.w, -0.1, 0.1)

    def forward(self, x):
        # x:[batch_size,seq_len]
        emb = self.embedding(x)  # [batch_size, seq_len, embedding_size]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * 2]

        M = self.tanh1(H)  # [batch_size, seq_len, hidden_size * 2]
        # 张量广播操作
        alpha = F.softmax(torch.matmul(M, self.w), dim=1)  # [batch_size, seq_len, 1]
        # 张量元素相乘，会发生张量广播使得张量的维度满足条件
        out = H * alpha  # [batch_size, seq_len, hidden_size * 2]
        # torch.sum操作默认情况下不保持维度
        out = torch.sum(out, 1)  # [batch_size,hidden_size * 2]
        out = self.tanh2(out)
        out = self.fc(out)  # [batch_size,num_classes]
        return out
