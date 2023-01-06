from BiLSTMattention import BiLSTMAttention
import pandas as pd
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
from train import train

def init_weights(m):
    """
    初始化模型
    :param m:
    :return:
    """
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)


def readData():
    # 读取数据
    comment_train = pd.read_csv('./data/comment_train_vec.csv')['Comment'].tolist()
    label_train = pd.read_csv('./data/label3_train.csv')['Star'].tolist()
    comment_test = pd.read_csv('./data/comment_test_vec.csv')['Comment'].tolist()
    label_test = pd.read_csv('./data/label3_test.csv')['Star'].tolist()


    print('数据读取成功！')

    # 将读取的数据中的字符串形式的列表转换成真正的列表
    def str_to_list(list):
        for i in range(len(list)):
            list[i] = eval(list[i])

    str_to_list(comment_train)
    str_to_list(comment_test)


if __name__ == '__main__':
    comment_train = []
    label_train = []
    comment_test = []
    label_test = []
    word2vec = None
    # 加载词向量模型
    if word2vec is None:
        word2vec = KeyedVectors.load_word2vec_format('./Word2Vec_model/data.vector')
        print('词向量加载成功！')
    # 使用GPU进行训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 建立模型
    model = BiLSTMAttention(
        vocab_size=len(word2vec.vectors),
        embed=word2vec.vectors,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.apply(init_weights)
    print('模型建立成功！')

    if len(comment_test) == 0:
        readData()

    # 定义学习率与迭代次数
    lr, num_epochs = 0.0001, 5
    trainer = torch.optim.Adam(model.parameters(), lr)  # 优化器
    loss = nn.CrossEntropyLoss().to(device)  # 损失函数

    train(model, trainer, loss, comment_train, label_train, comment_test, label_test, num_epochs, device)
