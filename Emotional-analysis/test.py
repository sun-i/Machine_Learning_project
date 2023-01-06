import torch
from Transformer import Transformer
from BiLSTM import Classifier
from TextCNN import TextCNN
from gensim.models import KeyedVectors
from BiLSTMattention import BiLSTMAttention
import pandas as pd

word2vec = None
data = []
label = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init():
    global word2vec
    global data
    global label
    if word2vec is None:
        word2vec = KeyedVectors.load_word2vec_format('./Word2Vec_model/data.vector')
        print('词向量加载成功！')
    data = pd.read_csv('./data/comment_test_vec.csv')['Comment'].tolist()
    data = data[:1000]
    label = pd.read_csv('./data/label3_test.csv')['Star'].tolist()
    label = label[:1000]
    str_to_list(data)
    print('数据加载成功！')

    # 将读取的数据中的字符串形式的列表转换成真正的列表


def str_to_list(list):
    for i in range(len(list)):
        list[i] = eval(list[i])


def evaluate(net, comments_data, labels_data):
    net.eval()
    sum_correct, i = 0, 0

    while i <= len(comments_data):
        # 批量计算正确率，一次计算64个评论信息
        comments = comments_data[i: min(i + 64, len(comments_data))]

        tokens_X = torch.tensor(comments).to(device=device)

        res = net(tokens_X)  # 获得到预测结果
        _, prediction = torch.max(res, 1)
        # 将预测结果从tensor转为array，并抽取结果
        prediction = prediction.cpu().numpy()
        for j in range(len(prediction)):
            print(prediction[j], '\t', labels_data[i + j])

        y = torch.tensor(labels_data[i: min(i + 64, len(comments_data))]).reshape(-1).to(device=device)

        sum_correct += (res.argmax(axis=1) == y).sum()  # 累加预测正确的结果
        i += 64

    return torch.true_divide(sum_correct, len(comments_data))  # 返回(总正确结果/所有样本)，精确率


if __name__ == '__main__':
    init()
    # 定义Bilstm-attention网络模型并初始化模型参数
    model1 = BiLSTMAttention(len(word2vec.vectors), word2vec.vectors)
    model1 = model1.to(device=device)
    model1.load_state_dict(torch.load('./model/BiLSTM-attention.parameters'))
    print('BiLSTM-attention模型加载成功！')
    acc = evaluate(model1, data, label)
    print('BiLSTM-attention模型准确率', acc)

    # 词嵌入维度为100，卷积核为[3, 4, 5], 输出通道数为[100, 100, 100]
    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    # 定义text-cnn网络模型并初始化模型参数
    model2 = TextCNN(len(word2vec.vectors), embed_size, kernel_sizes, nums_channels, word2vec.vectors)
    model2 = model2.to(device=device)
    model2.load_state_dict(torch.load('./model/best_text_cnn.parameters'))
    print('Text-CNN模型加载成功！')
    acc = evaluate(model2, data, label)
    print('Text-CNN模型准确率', acc)

    model3 = Classifier(vocab_size=len(word2vec.vectors), embed=word2vec.vectors, bidirectional=True)
    model3 = model3.to(device=device)
    model3.load_state_dict(torch.load('./model/BiLSTM.parameters'))
    print('BiLSTM模型加载成功！')
    acc = evaluate(model3, data, label)
    print('BiLSTM模型准确率', acc)

    model4 = Transformer(word2vec)
    model4 = model4.to(device=device)
    model4.load_state_dict(torch.load('./model/bext_transform.parameters'))
    print('Transformer模型加载成功！')
    acc = evaluate(model4, data, label)
    print('Transformer模型准确率', acc)
