import torch
from tqdm import tqdm

"""
评估函数，用以评估数据集在神经网络下的精确度
"""

def evaluate(net, comments_data, labels_data, device):
    sum_correct, i = 0, 0

    while i <= len(comments_data):
        # 批量计算正确率，一次计算64个评论信息
        comments = comments_data[i: min(i + 64, len(comments_data))]

        tokens_X = torch.tensor(comments).to(device=device)

        res = net(tokens_X)  # 获得到预测结果

        y = torch.tensor(labels_data[i: min(i + 64, len(comments_data))]).reshape(-1).to(device=device)

        sum_correct += (res.argmax(axis=1) == y).sum()  # 累加预测正确的结果
        i += 64

    return torch.true_divide(sum_correct, len(comments_data))  # 返回(总正确结果/所有样本)，精确率


"""
训练函数:用以训练模型，并保存最好结果时的模型参数
"""


def train(net, trainer, loss, train_comments, train_labels, test_comments, test_labels,
          num_epochs, device):
    # 启用模型的训练模式
    net.train()

    max_value = 0.5  # 初始化模型预测最大精度

    # 多次迭代训练模型
    for epoch in tqdm(range(num_epochs)):
        sum_loss, i = 0, 0  # 定义模型损失总和为 sum_loss, 变量 i
        pbar = tqdm(total=len(train_comments))
        while i < len(train_comments):
            # 批量64个数据训练模型
            comments = train_comments[i: min(i + 64, len(train_comments))]
            # X 转化为 tensor
            inputs_X = torch.tensor(comments).to(device=device)
            # Y 转化为 tensor
            y = torch.tensor(train_labels[i: min(i + 64, len(train_comments))]).to(device=device)
            # 将X放入模型，得到概率分布预测结果
            res = net(inputs_X)
            l = loss(res, y)  # 计算预测结果与真实结果损失
            trainer.zero_grad()  # 清空优化器梯度
            l.sum().backward()  # 后向传播
            trainer.step()  # 更新模型参数信息
            sum_loss += l.sum()  # 累加损失
            i += 16
            pbar.update(64)
        pbar.close()
        print('loss:\t', sum_loss / len(train_comments))
        # 计算训练集与测试集的精度
        train_acc = evaluate(net, train_comments, train_labels,device)
        test_acc = evaluate(net, test_comments, test_labels,device)
        # 保存下模型跑出最好的结果
        if test_acc >= max_value:
            max_value = test_acc
            torch.save(net.state_dict(), 'BiLSTM.parameters')
        # 输出训练信息
        print('-epoch:\t', epoch + 1,
              '\t-loss:\t', sum_loss / len(train_comments),
              '\ttrain-acc:', train_acc,
              '\ttest-acc:', test_acc)

