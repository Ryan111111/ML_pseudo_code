#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Ryan
# @Time: 2018/3/20 下午3:33

'''
K-means
通过迭代寻找K个聚类的一种划分方案，使得用这K个聚类的均值来代表各类样本时所得的总体误差最小
基础是最小误差平方和准则，各类内的样本越相似，其与该类均值间的误差平方越小，对所有类所得到的误差平方求和，即可验证分为k类，各聚类是否是最优的
#######################
创建k个点作为初始的质心点
当任意一个点的簇分配结果发生改变时：
   对数据集中每一个数据点：
      对每一个质心
         计算质心与数据点的距离
      将数据点分配到距离最近的簇
   对每一个簇，计算簇中所有点的均值，并将均值作为质心
#######################
'''
from numpy import *


#计算欧几里得距离
def EqualDistance(vector1,vector2):
    return sqrt(sum(power(vector2-vector1,2)))

#初始化聚类中心（这里是随机选择）
def initCentroids(train_data,k):
    numsamples,dim = train_data.shape
    centroids = zeros((k,dim))
    for i in range(k):
        index = int(random.uniform(0,numsamples))
        centroids[i,:] = train_data[index,:]
    return centroids    #返回K个聚类中心的点

def kmeans(train_data,k):
    numsamples = train_data.shape[0]
    clusterAssment = mat(zeros(numsamples,2))   #存储该样本所属集群,存储该样本与其质心之间的误差
    clusterChanged = True                       #质心改变标识位


    #step1: init centroids
    centroids = initCentroids(train_data,k)

    while clusterChanged:
        clusterChanged = False
        for i in range(numsamples):  #for each samples
            minDist = 100000.0
            minIndex = 0
            for j in range(k):       #for each centriods
                distance = EqualDistance(centroids[j,:],train_data[i,:])   #step2:找到最近的质心簇
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            #step3:update its cluster  更新其集群
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                clusterAssment[i,:] = minIndex,minDist**2

        #step4:update centriods
        for j in range(k):
            pointsInCluster = train_data[nonzero(clusterAssment[:,0].A == j)[0]]
            centroids[j,:] = mean(pointsInCluster,axis=0)

    return centroids,clusterAssment








