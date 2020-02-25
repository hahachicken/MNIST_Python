# coding: utf-8
import sys, os
sys.path.append(os.pardir)  #设置父文件目录
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

#导入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)#28*28像素 每个像素作为一个输入；0-9：10个输出

iters_num = 10000  #训练次数
train_size = x_train.shape[0]
batch_size = 100    #损失函数对象
learning_rate = 0.1 #学习率


iter_per_epoch = int(train_size / batch_size)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #计算梯度方向
    grad = network.numerical_gradient(x_batch, t_batch)
    
    #矩阵更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    
    if i % iter_per_epoch == 0:#输出   
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test) 
        print("Train_acc : Test_acc |", str(train_acc),":", str(test_acc))