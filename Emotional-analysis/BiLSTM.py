import torch
import torch.nn as nn

"""
    定义BiLSTM模型
"""


class Classifier(nn.Module):
    def __init__(self, vocab_size, embed, embedding_dim=100, hidden_dim=128,
                 n_layers=2, bidirectional=True, dropout=float(0.1)):
        super().__init__()

        # 定义embedding词嵌入模型，并将word2vec生成的词向量嵌入进来
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.from_pretrained(torch.tensor(embed))

        # lstm层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )

        # 全连接层
        num_direction = 2 if bidirectional else 1  # 双向&单向可选
        self.fc = nn.Linear(hidden_dim * n_layers * num_direction, 3)
        # 丢弃概率
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        # print('inputs ', inputs.shape)
        embedded = self.embedding(inputs)
        text_len = torch.tensor(inputs.shape[0] * [64])
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_len, batch_first=True)
        packed_output, (h_n, c_n) = self.lstm(packed_embedded)
        h_n = self.dropout(h_n)
        h_n = torch.transpose(h_n, 0, 1).contiguous()
        h_n = h_n.view(h_n.shape[0], -1)
        loggits = self.fc(h_n)

        return loggits
