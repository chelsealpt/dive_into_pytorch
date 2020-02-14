from d2lzh_pytorch import *
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
#生成数据
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
batch_size=10

features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
labels+=torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32)
print(features[0,:],labels[0])
#画出矢量图
set_figsize()
plt.scatter(features[:,1].numpy(),labels.numpy(),1)

num_examples = len(features)
indices = list(range(num_examples))
random.shuffle(indices)  # 样本的读取顺序是随机的
#生成模型参数并追踪
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True) 
#超参数
lr = 0.03
num_epochs = 3
#定义网络定义损失函数
net = linreg
loss = squared_loss


#用优化算法训练模型
for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失 
        #这里自定义的squared_loss是没有对batch维求平均，所以在sgd函数里除以了batch维
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
    
print(true_w, '\n', w)
print(true_b, '\n', b)

