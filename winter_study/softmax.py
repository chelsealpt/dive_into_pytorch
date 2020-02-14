'''
softmax回归简洁实现
PyTorch提供的函数往往具有更好的数值稳定性。
可以使用PyTorch更简洁地实现softmax回归。
'''
import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l
#设置批量大小，获取和读取数据，放进生成器
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10
#定义模型
class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x): # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y
#x形状 (batch_size, 1, 28, 28), 所以我们要先用view()将x的形状转换成(batch_size, 784)才送入全连接层
net = LinearNet(num_inputs, num_outputs)

#另一种简介定义模型方式
'''
from collections import OrderedDict

net = nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)
    '''
for param in net.parameters():
    print(param)

#初始化模型参数，我们使用均值为0、标准差为0.01的正态分布随机初始化模型的权重参数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0) 
#PyTorch提供了一个包括softmax运算和交叉熵损失计算的函数。它的数值稳定性更好
loss = nn.CrossEntropyLoss()
#定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
#训练模型
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

#epoch 1, loss 0.0031, train acc 0.749, test acc 0.757
#epoch 2, loss 0.0022, train acc 0.813, test acc 0.766
#epoch 3, loss 0.0021, train acc 0.825, test acc 0.808
#epoch 4, loss 0.0020, train acc 0.832, test acc 0.819
#epoch 5, loss 0.0019, train acc 0.836, test acc 0.824


X = torch.rand((2, 5))
X_prob = softmax(X)

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum()
    return X_exp / partition  # 这里应用了广播机制
X=torch.FloatTensor([100, 101, 102])
X_prob = softmax(X)