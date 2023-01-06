import pandas as pd
import re
import jieba
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

"""
    数据预处理模块
"""

stop_Words = []  # 存储停用词的列表


def drop_meaningless(comment):
    """
    去除无用字符

    :param comment: 评论文本
    :return: 去除无用字符后的评论文本
    """
    pattern1 = '[a-zA-Z0-9]'
    pattern2 = re.compile(u'[^\s1234567890:：' + '\u4e00-\u9fa5]+')
    pattern3 = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    line1 = re.sub(pattern1, '', comment)  # 去除英文字母和数字
    line2 = re.sub(pattern2, '', line1)  # 去除表情和其他字符
    line3 = re.sub(pattern3, '', line2)  # 去除去掉残留的冒号及其它符号
    new_comment = ''.join(line3.split())  # 去除空白
    return new_comment


def comment_cut(comment):
    """
    jieba 分词
    :param comment: 评论文本
    :return: 分词后的列表
    """
    return list(jieba.cut(comment))


def drop_stopWords(comment):
    """
    去除停用词

    :param comment: 评论文本
    :return: 去除停用词之后的列表
    """
    # 加载停用词表
    if len(stop_Words) == 0:
        with open('./stopWords/baidu_stopwords.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for word in lines:
                word = word.strip()  # strip 去掉\n换行符
                stop_Words.append(word)
    return [word for word in comment if word not in stop_Words]


def cut_and_padding(comment):
    """
    对评论进行截断和填充的函数

    :param comment:
    :return:
    """
    if len(comment) < 64:
        # 若长度未能达到 64 时，进行填充操作
        comment += (64 - len(comment)) * ['<pad>']
    else:
        # 若长度超出 64，进行截断操作
        comment = comment[:64]

    # 返回填充或截取后文本
    return comment


def preProcessData(data):
    """
    数据预处理函数

    :param data: 数据
    :return:
    """
    data["Comment"] = data['Comment'].apply(drop_meaningless)
    data['Comment'] = data['Comment'].apply(comment_cut)
    data['Comment'] = data['Comment'].apply(drop_stopWords)
    data['Comment'] = data['Comment'].apply(cut_and_padding)
    if 'Star' in data:
        for i in range(data['Star'].size):
            if data['Star'][i] == 1 or data['Star'][i] == 2:
                data['Star'][i] = 0
            elif data['Star'][i] == 4 or data['Star'][i] == 5:
                data['Star'][i] = 2
            else:
                data['Star'][i] = 1


def word2vecTrain(data):
    """
    训练词向量模型
    :param data:
    :return:
    """
    # 将 Comment 列转换成列表，便于 word2vec 模型处理
    comments = data['Comment'].tolist()
    # 训练 Word2Vec 模型，将词转换为向量
    model = Word2Vec(
        comments,
        sg=0,
        vector_size=100,
        window=5,
        min_count=1,
    )
    # 词向量保存
    model.wv.save_word2vec_format('./Word2Vec_model/data.vector', binary=False)
    # 模型保存
    model.save('./Word2Vec_model/test.model')

    return model.wv


def corpus(comment, model):
    """
    将文本转换成向量形式
    :param comment: 文本
    :param model: 词向量模型
    :return:
    """
    comment_list = []
    for i in comment:
        try:
            comment_list.append(model.get_index(i))
        except:
            comment_list.append(0)
    return comment_list


def cut_train_test(data):
    comment = data['Comment']
    label = data['Star']
    # 80%为训练集。20%为测试集
    comment_train, comment_test, label_train, label_test = train_test_split(comment, label, test_size=0.2,
                                                                            random_state=1000)
    print(comment_train.head(), label_train.head())

    # 将训练集和测试集存入到文件中
    train_data_path = './data/train/'
    test_data_path = './data/test/'
    comment_train.to_csv(train_data_path + 'comment_train_vec.csv')
    comment_test.to_csv(test_data_path + 'comment_test_vec.csv')
    label_train.to_csv(train_data_path + 'label_train_vec.csv')
    label_test.to_csv(test_data_path + 'label_test_vec.csv')


if __name__ == '__main__':
    douban_comment_data = pd.read_csv('./data/douban comments.csv')
    data = douban_comment_data[['Comment', 'Star']]
    preProcessData(data)
    model = word2vecTrain(data)
    data['Comment'] = data['Comment'].apply(corpus, args=(model,))
    cut_train_test(data)
