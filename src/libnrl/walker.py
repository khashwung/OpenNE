from __future__ import print_function
import random
import numpy as np
import multiprocessing

def deepwalk_walk_wrapper(class_instance, walk_length, start_node):
    class_instance.deepwalk_walk(walk_length, start_node)

class BasicWalker:
    def __init__(self, G, workers):
        self.G = G.G
        self.node_size = G.node_size
        self.look_up_dict = G.look_up_dict

    def deepwalk_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        look_up_dict = self.look_up_dict
        node_size = self.node_size

        # 初始化第一个节点
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            # G.neighbors，networkx的一个方法，获取节点的neighbor views，然后转化为list
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                # 随机选择一个邻居，作为当前walk的节点
                walk.append(random.choice(cur_nbrs))
            else:
                # 当前节点没有邻居，结束。这种情况应该不存在，因为肯定存在一条跳过来的节点。但是可能存在往回跳的情况，比如单线
                # 图，达到尾点，又沿着来的路往回走
                break
        # 返回一个walk list
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        # 所有节点，并转化为list，每个节点都会作为起始节点walk一遍
        nodes = list(G.nodes())
        print('Walk iteration:')
        # walk num_walks次，每次都以所有的节点为起始点进行walk
        for walk_iter in range(num_walks):
            # pool = multiprocessing.Pool(processes = 4)
            print(str(walk_iter+1), '/', str(num_walks))
            # 每个epoch都重新shuffle
            random.shuffle(nodes)
            for node in nodes:
                # walks.append(pool.apply_async(deepwalk_walk_wrapper, (self, walk_length, node, )))
                walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=node))
            # pool.close()
            # pool.join()
        # print(len(walks))
        # 这个walks包含了多个epoch
        return walks


class Walker:
    def __init__(self, G, p, q, workers):
        self.G = G.G
        self.p = p
        self.q = q
        self.node_size = G.node_size
        self.look_up_dict = G.look_up_dict

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        look_up_dict = self.look_up_dict
        node_size = self.node_size

        walk = [start_node]

        # 不满足长度就要不断的walk
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    # 如果只有一个节点，那么还没有可以进行LINE特有的walk方法的边，因此采用普通walk
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    # 已经有了至少两个节点了，可以构成一个长度为1的路径，采用LINE特有的walk方法
                    prev = walk[-2]
                    pos = (prev, cur)
                    next = cur_nbrs[alias_draw(alias_edges[pos][0], 
                        alias_edges[pos][1])]
                    walk.append(next)
            else:
                break

        # 完成了个点固定长度的采样
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        # walk num_walks epoches
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            # 每个epoch需要shuffle
            random.shuffle(nodes)
            # 遍历所有的节点
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        # 这里有三个类型的点：src，dst，dst的邻居（三种情况：1.为src，为和src相连的点，为不和src相连的点）
        # 不同类型的边，处理方式不一样，但是最终都要加在一起进行归一化
        for dst_nbr in G.neighbors(dst):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        # 还是生成alias表和prob表，为了进行采样
        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        初始化时就要做，主要用于初始化alias_nodes和alias_edges
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G

        alias_nodes = {}
        # 遍历所有节点，对每个节点都构建用于multinomial分布采样的alias table和prob table
        for node in G.nodes():
            # 提取出当前节点所有的邻居权重
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in G.neighbors(node)]
            # 计算归一化因子，为当前节点的邻居权重之和
            norm_const = sum(unnormalized_probs)
            # 归一化
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            # 对当前节点构建一个alias table和prob table
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        look_up_dict = self.look_up_dict
        node_size = self.node_size
        # 遍历所有的边
        for edge in G.edges():
            # 对于每个边，都要计算其dst的归一化跳转概率
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        # alias_nodes为达到某一节点后，选择哪条边的alias setup
        # alias_edges为达到某一边后，选择哪个
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    Multinomial分布的快速采样。时间复杂度为o(n)。生成两个list，一个Prob table，一个alias table。
    该方法本质上是将multinomial分布转换为了一个uniform分布+一个二元分布
    setup首先是构建alias table和prob table，为后面的alias draw采样做准备
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    # 长度为K的index信息
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # 不断迭代，直到smaller和larger中没有值了
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    # 返回的是一个tuple (J, q)
    return J, q

def alias_draw(J, q):
    '''
    根据alias table和prob table。进行两次采样。第一次uniform分布采样，第二次是binomial分布采样
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
