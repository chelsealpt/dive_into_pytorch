import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l
from d2lzh_pytorch import *
#加载数据，放进生成器并打乱顺序
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs = 784
num_outputs = 10
#生成模型参数并追踪
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
#定义模型 通过view函数将每张原始图像改成长度为num_inputs的向量 转换成(batch_size,num_inputs)
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
#定义损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))
#超参数
num_epochs, lr = 5, 0.1

#用优化算法训练模型
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
#epoch 1, loss 0.7842, train acc 0.749, test acc 0.792
#epoch 2, loss 0.5698, train acc 0.815, test acc 0.810
#epoch 3, loss 0.5267, train acc 0.825, test acc 0.819
#epoch 4, loss 0.5009, train acc 0.833, test acc 0.825
#epoch 5, loss 0.4848, train acc 0.837, test acc 0.828

#画图
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])



