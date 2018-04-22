#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Ryan
# @Time: 2018/3/20 下午12:38

'''
KNN:如果一个样本在特征空间中k个最相似的样本中的大多数属于一个类别，则该样本也属于这个类别

########################
计算已知类别数据集中的点与当前点之间的距离
按照距离递增次序排序
选取与当前点距离最小的k个点
确定前k个点所在类别的出现频率
返回前k个点出现频率最高的类别作为当前点的预测类别
'''

import numpy as np
import operator

def KNN(test_data,train_data,labels,k):
    '''
    :param test data
    :param train data
    :param labels: labels
    :param k: params for KNN 选择距离最小的K个点
    :return:
    '''
    train_samples_num = train_data.shape[0]

    #caluate testsamples to each trainsamples distance  计算测试数据导每个训练样本的距离
    diffmat = np.tile(test_data,(train_samples_num,1))-train_data      #测试样本，重复train_samples_num行，构成与训练样本一样的矩阵，直接相减就可以得到该样本与每一个训练数据的距离
    #np.tile()相当于在列向量上面重复，广播的意思
    sqdiffmat = diffmat**2
    sqdistances = sqdiffmat.sum(axis=1)  #把每一行的距离加起来
    distances = sqdistances**0.5

    #return distance index of min to max  返回distance元素中从小到大排序后的索引值
    sortedDistindex = distances.argsort()
    classcount = {}

    #get the top of k element
    for i in range(k):
        votelabel = labels[sortedDistindex[i]]
        classcount[votelabel] = classcount.get(votelabel,0) + 1   #计算出类别次数

    #对类别次数进行排序  the max voted class will return
    sortedClassCount = sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    #key=operator.itemgetter(1)根据字典的值进行排序
    return sortedClassCount[0][0]  #取字段排序后的第一行的键


