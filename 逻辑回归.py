#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Ryan
# @Time: 2018/3/20 上午11:24

'''
求解代价函数：
1 优化函数存在解析解，求导之后就可以得到最优解
2 式子非常难以求导，函数里面存在隐含的变量或变量互相存在耦合；求导之后得不到解析解；未知参数大于方程组个数
  这个时候就需要我们一步一步迭代找到最优解
3 如果函数是凸函数，那么就存在全局最优解，但是如果是非凸函数，那么存在局部最优解

注意：一般梯度下降算法是在每次更新回归系数的时候都需要遍历整个数据集
     改进随机梯度下降：一次仅用一个样本点来更新回归系数

随机梯度下降的伪代码：
#####################
初始化回归系数1
重复下面步骤直到收敛{
对随机遍历的数据集中的每个样本
    随着迭代的逐渐进行，减少alpha的值
   计算该样本的梯度
   使用alpha x gradient来更新回归系数
}
返回回归系数值
#####################
'''
from numpy import *


#定义sigmoid函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#train a logistic regression model using some optional optimize algorithm
def trainLogRegres(train_x,train_y,opts):
    '''
    :param train_x: train_x is a mat datatype,each row stands for one sample
    :param train_y: train_y is a mat datatype,each row is the corresonding label
    :param opts: opts is optimize option include step and maximum number of iteration
    :return:
    '''
    numSamples,numFeatures = shape(train_x)
    alpha = opts['alpha']
    maxIter = opts['maxIter']
    weights = ones((numFeatures,1))

    #optimize through gradient descent algorilthm 通过梯度下降算法进行优化
    for k in range(maxIter):
        if opts['optimizeType'] == 'gradDescent':
            output = sigmoid(train_x*weights)    #initialze regression for all samples
            error = train_y-output
            weights = weights+alpha*train_x.transpose()*error #trainspose()转置，updates weights

        if opts['optimizeType'] == 'stocgradDescent':  #随机梯度下降
            for i in range(numSamples):                #通过某一个样本更新权重
                output = sigmoid(train_x[i,:]*weights)
                error = train_y[i,0]-output
                weights = weights + alpha*train_x[i,:].transpose()*error
        return weights

def testLogRegression(weights,test_x,test_y):
    '''
    test you trained model
    :param weights:
    :param test_x:
    :param test_y:
    :return: accuracy
    '''
    numSamples,numFeatures = shape(test_x)
    matchcount = 0  #count correct num
    for i in range(numSamples): #for each samples
        predict = sigmoid(test_x[i,:]*weights)[0,0] > 0.5   #if predict >0.5 is 1
        if predict ==bool(test_y[i,0]):
            matchcount +=1
    accuracy = float(matchcount)/numSamples
    return accuracy









