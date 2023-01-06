import torch
import torch.nn as nn

"""
定义Text-CNN模型
"""
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, embed, **kwargs):
        super(TextCNN, self).__init__(**kwargs)

        # 定义embedding词嵌入模型，并将word2vec生成的词向量嵌入进来
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.from_pretrained(torch.tensor(embed))

        # 此嵌入层同上
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.constant_embedding.from_pretrained(torch.tensor(embed))

        # 减少模型复杂度，并定义全连接层decoder
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 3)

        # 池化层
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()

        # 创建多个一维卷积层
        self.convs = nn.ModuleList()

        # 追加多个一维卷积层，用以提取文本的不同特征信息
        for c, k in zip(num_channels, kernel_sizes):
            # 输入通道数为 2 * embed_size, 输出通道数为 c， 卷积核为 k
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # 沿着向量维度将两个嵌入层连结起来，
        # 每个嵌入层的输出形状都是（批量大小，词元数量，词元向量维度）连结起来
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)

        # 根据一维卷积层的输入格式，重新排列张量，以便通道作为第2维
        embeddings = embeddings.permute(0, 2, 1)

        # 每个一维卷积层在池化层合并后，获得的张量形状是（批量大小，通道数，1）
        # 删除最后一个维度并沿通道维度连结
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)

        # 使用全连接层输出概率分布，并返回概率结果
        outputs = self.decoder(self.dropout(encoding))
        return outputs

