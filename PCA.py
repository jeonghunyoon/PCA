# -*- coding: utf-8 -*-
__author__ = "Jeonghun Yoon"
import numpy as np
import matplotlib.pyplot as plot

### 1. Load data form test data.
f = open("test.txt")

data = []
for line in f.readlines():
    data.append(map(float, line.strip().split("\t")))


def pca(data, topNFeat=9999999):
    # axis = 0 일 때는 컬럼의 평균을 구한다. 1 일 때는 로우의 평균을 구한다.
    meanVal = np.mean(data, axis=0)
    stdVal = np.std(data, axis=0)
    # x를 normalize 시킨다. 평균을 빼고, 표준편차를 나눈다.
    normalData = data - meanVal
    normalData = normalData / stdVal

    # covariance matrix
    covMat = np.cov(normalData, rowvar=False)

    # get eigenvectors and eigenvalues
    eigenVals, eigenVecs = np.linalg.eig(np.mat(covMat))

    # principal component는 가장 큰 eigenvalue와 asso 된 eigenvector이다
    eigenValsIdx = np.argsort(eigenVals)

    # eigenvector와 eigenvalue를 sorting 한다. ::-1은 reverse and column
    eigenVecs = eigenVecs[eigenValsIdx[::-1]]
    removedEigenVecs = eigenVecs[:,range(topNFeat)]

    # removed data
    removedData = normalData * removedEigenVecs

    # removed data -> original domain ()
    recovData = (np.multiply(removedData * removedEigenVecs.T, stdVal)) + meanVal

    return removedData, recovData


removedData, recovData = pca(data, 1)

fig = plot.figure()
ax = fig.add_subplot(111)
ax.scatter(np.mat(data)[:,0].flatten().A[0], np.mat(data)[:,1].flatten().A[0], marker="^", s=90)
ax.scatter(recovData[:,0].flatten().A[0], recovData[:,1].flatten().A[0], marker="o", s=50, c="red")
plot.show()