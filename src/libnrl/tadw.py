from __future__ import print_function
import math
import numpy as np
from numpy import linalg as la
from sklearn.preprocessing import normalize
from .gcn.utils import *

class TADW(object):
    
    def __init__(self, graph, dim, lamb=0.2):
        self.g = graph
        self.lamb = lamb
        self.dim = dim
        self.train()

    def getAdj(self):
        graph = self.g.G
        node_size = self.g.node_size
        look_up = self.g.look_up_dict
        # 这是一个O(n^2)的空间开辟，因此应该该算法不适用于大数据集
        adj = np.zeros((node_size, node_size))
        for edge in self.g.G.edges():
            # 如果两点之间有边，就沿着其方向将矩阵值设为1，因为是无向图，所以对称设置
            adj[look_up[edge[0]]][look_up[edge[1]]] = 1.0
            adj[look_up[edge[1]]][look_up[edge[0]]] = 1.0
        # ScaleSimMat
        # 归一化，沿着当前节点的度数进行归一化
        return adj/np.sum(adj, axis=1)

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.dim))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,' '.join([str(x) for x in vec])))
        fout.close()

    def getT(self):
        g = self.g.G
        look_back = self.g.look_back_list
        # 将所有的节点特征拼接起来
        self.features = np.vstack([g.nodes[look_back[i]]['feature']
            for i in range(g.number_of_nodes())])
        # pca降维表示
        self.preprocessFeature()
        return self.features.T

    def preprocessFeature(self):
        # 对文本信息进行svd分解，并去除噪声，重建文本信息矩阵
        U, S, VT = la.svd(self.features)
        Ud = U[:, 0:200]
        # 文档中说了，s为能量强度，按照降序排列，因此取前200个强度的向量即可
        Sd = S[0:200]
        # 一个疑点，为啥要这样构建文本特征，而不是再乘以VT
        # 其实是pca的降维映射。。。忘了，pca'沿着节点的维度降维，每个节点的文本特征是200维
        # [[1,2],[3,4]] * [2,4] = np.dot([[1,2],[3,4]],np.diagflat([2,4])) = [[2,8],[6,16]]
        self.features = np.array(Ud)*Sd.reshape(200)

    def train(self):
        self.adj = self.getAdj()
        # M=(A+A^2)/2 where A is the row-normalized adjacency matrix
        # 构建M矩阵，M矩阵
        self.M = (self.adj + np.dot(self.adj, self.adj))/2
        # T is feature_size*node_num, text features
        self.T = self.getT()
        self.node_size = self.adj.shape[0]
        self.feature_size = self.features.shape[1]
        # 这是要学习的参数：W，H。先随机初始化
        self.W = np.random.randn(self.dim, self.node_size)
        self.H = np.random.randn(self.dim, self.feature_size)
        # Update
        for i in range(20):
            print('Iteration ', i)
            # 有点Coordinate Descent的意思，固定某一个，优化另一个参数
            # Update W
            B = np.dot(self.H, self.T)
            # 这步是 dLoss/dW
            drv = 2 * np.dot(np.dot(B, B.T), self.W) - \
                    2*np.dot(B, self.M.T) + self.lamb*self.W
            Hess = 2*np.dot(B, B.T) + self.lamb*np.eye(self.dim)
            # 将梯度变为一维向量，便于计算理解
            drv = np.reshape(drv, [self.dim*self.node_size, 1])
            # 将梯度方向反向为下降方向
            rt = -drv
            dt = rt
            # 为了方便理解，将所有的参数都变成一维向量
            vecW = np.reshape(self.W, [self.dim*self.node_size, 1])
            # 收敛条件：梯度已经接近为0
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.node_size))
                Hdt = np.reshape(np.dot(Hess, dtS), [self.dim*self.node_size, 1])

                at = np.dot(rt.T, rt)/np.dot(dt.T, Hdt)
                # 可以看到，其实是梯度下降法
                vecW = vecW + at*dt
                rtmp = rt;
                # 这里应该是更新动量之类的
                rt = rt - at*Hdt
                bt = np.dot(rt.T, rt)/np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.W = np.reshape(vecW, (self.dim, self.node_size))

            # Update H
            drv = np.dot((np.dot(np.dot(np.dot(self.W, self.W.T),self.H),self.T)
                    - np.dot(self.W, self.M.T)), self.T.T) + self.lamb*self.H
            drv = np.reshape(drv, (self.dim*self.feature_size, 1))
            rt = -drv
            dt = rt
            vecH = np.reshape(self.H, (self.dim*self.feature_size, 1))
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.feature_size))
                Hdt = np.reshape(np.dot(np.dot(np.dot(self.W, self.W.T), dtS), np.dot(self.T, self.T.T))
                                + self.lamb*dtS, (self.dim*self.feature_size, 1))
                at = np.dot(rt.T, rt)/np.dot(dt.T, Hdt)
                # 同理，也是梯度下降
                vecH = vecH + at*dt
                rtmp = rt
                rt = rt - at*Hdt
                bt = np.dot(rt.T, rt)/np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.H = np.reshape(vecH, (self.dim, self.feature_size))
        # 将三个embedding矩阵放在一起
        self.Vecs = np.hstack((normalize(self.W.T), normalize(np.dot(self.T.T, self.H.T))))
        # get embeddings
        self.vectors = {}
        look_back = self.g.look_back_list
        for i, embedding in enumerate(self.Vecs):
            self.vectors[look_back[i]] = embedding




