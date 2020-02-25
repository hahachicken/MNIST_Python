import sys, os
sys.path.append(os.pardir)  #设置父文件目录：用于访问dataset
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

#本文件实现了多层次网络对象

class MultiLayerNet:
    """
    属性量：
     input_size：输入大小（MNIST为784）
     hidden_size_list：隐藏层神经元数量列表（例如[100,100,100]）
     output_size：输出大小（MNIST为10）
     激活函数：'relu'或'sigmoid'
     weight_init_std：指定重量的标准偏差（例如0.01）
         如果指定'relu'或'he'，设置He初始”
         如果指定'sigmoid'或'xavier'，设置Xavier初始值
     weight_decay_lambda：重量衰减强度
     """

    def __init__(self, input_size, hidden_size_list, output_size = 10,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.params = {}
        self.weight_decay_lambda = weight_decay_lambda

        # 权重初始化
        self.__init_weight(weight_init_std)

        # 生成隐藏层
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
            self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """重量的初始值设定

        变量：weight_init_std：指定重量的标准偏差（例如0.01）
             如果指定'relu'或'he'，设置“2/√n”
             如果指定'sigmoid'或'xavier'，设置“1/√n”
        
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU时的建议初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid时的建议初始值

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """找到损失函数

        变量：
         x：输出数据
         t：正确标签

        返回：
         损失函数值
        """
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """求梯度（数值法求导数）

        变量：
         x：输入数据
         t：教师标签

        返回
         字典变量表示每层的变量
             grads ['W1']，grads ['W2']		是每层的权重
             grads ['b1']，grads ['b2']		是每层的偏置
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """计算梯度

        变量：
         x：输入数据
         t：教师标签

        返回
         字典变量与每层的渐变
             grads ['W1']，grads ['W2']		是每层的权重
             grads ['b1']，grads ['b2']		是每层的偏置
        """

        # forward
        self.loss(x, t)

        # backward
        dout = 1 #输出层对自己偏导=1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)


        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
