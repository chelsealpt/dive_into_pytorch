import torch
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
#读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256
#定义模型参数
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)
#定义激活函数
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))
#定义模型结构
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2
#定义损失函数
loss = torch.nn.CrossEntropyLoss()
#训练模型
#这里学习率这么大，是因为用的是pytorch的损失函数（已对batch维求平均），而用的优化器是自定义的sgd（又对batch维求了平均，所以这里要对学习率调成原来batch_size倍
num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
#epoch 1, loss 0.0030, train acc 0.712, test acc 0.791
##epoch 2, loss 0.0019, train acc 0.823, test acc 0.817
#epoch 3, loss 0.0017, train acc 0.844, test acc 0.831
#epoch 4, loss 0.0015, train acc 0.854, test acc 0.843
#epoch 5, loss 0.0015, train acc 0.864, test acc 0.857
