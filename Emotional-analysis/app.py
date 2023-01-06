import json
import os
from Preprocess import preProcessData
from Preprocess import corpus
import pandas as pd
from flask import *
import torch
import Preprocess as pre
from gensim.models import KeyedVectors
from BiLSTMattention import BiLSTMAttention

app = Flask(__name__)
model = None
word2vec = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT_PATH = './data'


@app.route('/')
def init():  # put application's code here
    global model
    global word2vec
    if word2vec is None:
        word2vec = KeyedVectors.load_word2vec_format('./Word2Vec_model/data.vector')
        print('词向量加载成功！')

    # 定义text-cnn网络模型并初始化模型参数
    model = BiLSTMAttention(len(word2vec.vectors), word2vec.vectors)
    model = model.to(device=device)
    model.load_state_dict(torch.load('./model/BiLSTM-attention.parameters'))
    print('模型加载成功！')
    model.eval()
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    global word2vec
    # 获取前端json数据
    data = request.get_data()
    json_data = json.loads(data)
    print(json_data)
    text = json_data['text']
    text = pre.drop_meaningless(text)
    text = pre.comment_cut(text)
    text = pre.drop_stopWords(text)
    print(text)
    text = pre.cut_and_padding(text)
    text = pre.corpus(text, word2vec)
    text = [text]
    print(text)
    inputs_X = torch.tensor(text).to(device)
    output = model(inputs_X)
    _, prediction = torch.max(output, 1)
    # 将预测结果从tensor转为array，并抽取结果
    prediction = prediction.cpu().numpy()[0]
    result = None
    if prediction == 0:
        result = '差评'
    elif prediction == 1:
        result = '中评'
    else:
        result = '好评'
    info = dict()
    print(result)
    if result is None:
        info['status'] = 'error'
    else:
        info['status'] = 'success'
    info['result'] = result
    return jsonify(info)


@app.route('/Single', methods=['POST'])
def Single():
    global word2vec
    file = request.files['file']
    file.save(os.path.join(ROOT_PATH, file.filename))
    data = pd.read_csv(os.path.join(ROOT_PATH, file.filename), encoding='gbk')
    preProcessData(data)
    data['Comment'] = data['Comment'].apply(corpus, args=(word2vec,))
    input = data['Comment'].tolist()
    inputs_X = torch.tensor(input).to(device)
    output = model(inputs_X)
    _, prediction = torch.max(output, 1)
    # 将预测结果从tensor转为array，并抽取结果
    prediction = prediction.cpu().numpy()
    print('预测完成')
    result = 0
    for i in prediction:
        if i == 0:
            result += 10
        elif i == 1:
            result += 8
        else:
            result += 5
    result = result * 1.0 / len(prediction)
    info = dict()
    info['status'] = 'success'
    info['result'] = str(result)[:4]
    return info


if __name__ == '__main__':
    app.run()
