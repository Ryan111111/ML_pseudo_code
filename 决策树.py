#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Ryan
# @Time: 2018/3/20 下午2:17


'''
信息增益 = 经验熵（类别熵）-条件经验熵（特征熵）
'''
from math import log

def calcshannonEnt(train_data):
    '''
    计算经验熵，香农熵
    :param dataset:
    :return:
    '''
    train_samples_num = len(train_data)
    labelCounts = {}   #保存每个标签出现的次数
    for featureCount in train_data:  #对每组特征向量进行统计
        currentLabel = featureCount[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1

    #计算香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/train_samples_num
        shannonEnt -=prob*log(prob,2)

    return shannonEnt


def splitDataSet(train_data,axis,value):
    '''
    按照给定特征划分数据集
    :param train_data:
    :param axis:
    :param value:
    :return:
    '''
    retDataSet = []   #创建返回的数据集列表
    for featVec in train_data:
        if featVec[axis] == value:
            #将符合的特征数据取出
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(train_data):
    '''
    选择最优特征
    :param train_data:
    :return:最优特征索引值
    '''
    train_samples_num = len(train_data[0])-1   #特征的数量
    baseEntropy = calcshannonEnt(train_data)   #计算数据集的经验熵
    bestInfoGain = 0.0   #信息增益
    beatFeature = -1     #最优特征索引值
    for i in range(train_samples_num):   #遍历所有特征
        featList = [example[i] for example in train_data]
        uniqueVals = set(featList)
        newEntropy = 0.0            #经验条件熵
        for value in uniqueVals:    #计算信息增益
            subDataset = splitDataSet(train_data,i,value)   #划分后的子集
            prob = len(subDataset)/float(len(train_data))   #获取某个特征所占相应的概率
            newEntropy += prob*calcshannonEnt(subDataset)   #根据公式计算经验条件熵，记得每个经验熵需要乘以对应的概率
        infoGain = baseEntropy-newEntropy                   #信息增益
        print('第%d个特征的信息增益为%.3f'%(i,infoGain))       #打印每个特征的信息增益

        #获取信息增益最大的特征下标
        if (infoGain>bestInfoGain):
            bestInfoGain = infoGain
            beatFeature = i

    return beatFeature



