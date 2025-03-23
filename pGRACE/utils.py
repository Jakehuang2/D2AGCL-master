import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv
from torch_geometric.utils import degree, to_networkx
from torch_scatter import scatter
import networkx as nx
import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_base_model(name: str):
    def gat_wrapper(in_channels, out_channels):
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            heads=4
        )

    def gin_wrapper(in_channels, out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ELU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        return GINConv(mlp)

    base_models = {
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GraphConv': GraphConv,
        'GINConv': gin_wrapper
    }

    return base_models[name]


def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]


def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x


def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)

def generate_split_sequential(num_samples: int, train_ratio: float, val_ratio: float):
    # 计算训练集、验证集和测试集的大小
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    # 按照顺序切分数据集
    train_set = torch.arange(0, train_len)
    val_set = torch.arange(train_len, train_len + val_len)
    test_set = torch.arange(train_len + val_len, num_samples)

    # 创建掩码
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    # 将对应的索引设置为True
    train_mask[train_set] = True
    test_mask[test_set] = True
    val_mask[val_set] = True

    return train_mask, test_mask, val_mask
def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    # 生成顺序索引
    indices = torch.arange(0, num_samples)

    # 按顺序划分
    idx_train = indices[:train_len]
    idx_val = indices[train_len:train_len + val_len]
    idx_test = indices[train_len + val_len:]
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask
def generate_split_random(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask

def show_perturbed_edge(data,perturbed_edges):
    # 创建6个集合
    train_train_edges = []
    train_val_edges = []
    train_test_edges = []
    val_val_edges = []
    val_test_edges = []
    test_test_edges = []
    others = []
    dif_label_num = 0
    labels = data.y
    for edge in perturbed_edges[::2]:
        node1, node2 = edge[0], edge[1]
        if (labels[node1].item() != labels[node2].item()):
            dif_label_num += 1
        if data.train_mask[node1] and data.train_mask[node2]:
            train_train_edges.append(edge)
        elif (data.train_mask[node1] and data.val_mask[node2]) or (
                data.train_mask[node2] and data.val_mask[node1]):
            train_val_edges.append(edge)
        elif (data.train_mask[node1] and data.test_mask[node2]) or (
                data.train_mask[node2] and data.test_mask[node1]):
            train_test_edges.append(edge)
        elif data.val_mask[node1] and data.val_mask[node2]:
            val_val_edges.append(edge)
        elif (data.val_mask[node1] and data.test_mask[node2]) or (
                data.val_mask[node2] and data.test_mask[node1]):
            val_test_edges.append(edge)
        elif data.test_mask[node1] and data.test_mask[node2]:
            test_test_edges.append(edge)
        else:
            others.append(edge)

    print(f'Train-Train edges: {len(train_train_edges)}')
    print(f'Train-Val edges: {len(train_val_edges)}')
    print(f'Train-Test edges: {len(train_test_edges)}')
    print(f'Val-Val edges: {len(val_val_edges)}')
    print(f'Val-Test edges: {len(val_test_edges)}')
    print(f'Test-Test edges: {len(test_test_edges)}')
    print(f'Others: {len(others)}')
    print(f'dif_label_num: {dif_label_num}')
    # print(len(perturbed_edges))
    # print(perturbed_edges)
    print(torch.where(data.train_mask)[0])
def getCandidateEdge2(data, adj, idx_train):

    print("候选边生成中")
    idx_val = torch.where(data.val_mask)[0]
    idx_test = torch.where(data.test_mask)[0]

    idx_unlabeled = torch.cat((idx_val, idx_test)).unique()
    # idx_unlabeled = idx_test


    # 计算每个节点的度数
    node_degreed = data.edge_index[0].bincount(minlength=data.num_nodes)
    # 计算图中节点的平均度数
    avg_degree = node_degreed.float().mean()

    # 筛选度数小于平均度数的训练集和测试集节点
    low_degree_train_nodes = idx_train[node_degreed[idx_train] < avg_degree]
    low_degree_test_nodes = idx_unlabeled[node_degreed[idx_unlabeled] < avg_degree]
    # print(low_degree_test_nodes)
    # print(low_degree_train_nodes)
    # 预先筛选出符合条件的节点对
    candidate_train_edges = []
    candidate_test_edges = []
    for row in low_degree_train_nodes:
        for column in low_degree_train_nodes:
            if row != column and adj[row, column] == 0:
                candidate_train_edges.append((row, column))

    for row in low_degree_train_nodes:
        for column in low_degree_test_nodes:
            if adj[row, column] == 0:
                candidate_test_edges.append((row, column))
    candidate_train_adj_mask = torch.zeros_like(adj, dtype=torch.bool)
    candidate_test_adj_mask = candidate_train_adj_mask.clone()
    # 将 candidate_edges 的边标记为 True
    for row, column in candidate_train_edges:
        candidate_train_adj_mask[row, column] = True
    for row, column in candidate_test_edges:
        candidate_test_adj_mask[row, column] = True
    # 如果是无向图，可以对称设置
    candidate_train_adj_mask = candidate_train_adj_mask | candidate_train_adj_mask.T
    candidate_train_adj_mask_1d = candidate_train_adj_mask.view(-1)
    candidate_test_adj_mask = candidate_test_adj_mask | candidate_test_adj_mask.T
    candidate_test_adj_mask_1d = candidate_test_adj_mask.view(-1)
    print("候选边生成结束")
    return candidate_train_adj_mask_1d,candidate_test_adj_mask_1d


def getCandidateEdge(data, adj, idx_train, idx_unlabeled):
    print("候选边生成中")
    # 计算每个节点的度数
    node_degreed = data.edge_index[0].bincount(minlength=data.num_nodes)
    # 计算图中节点的平均度数
    avg_degree = node_degreed.float().mean()

    # 筛选度数小于平均度数的训练集和测试集节点
    low_train_degree_nodes = idx_train[node_degreed[idx_train] < avg_degree]
    low_test_degree_nodes = idx_unlabeled[node_degreed[idx_unlabeled] < avg_degree]

    # 确定低度节点的集合
    train_nodes = low_train_degree_nodes
    test_nodes = low_test_degree_nodes

    # 确保生成的候选边：一个节点来自 train_nodes，另一个来自 test_nodes
    rows = train_nodes.repeat_interleave(test_nodes.size(0))  # 每个 train 节点重复 test_nodes.size(0) 次
    cols = test_nodes.repeat(train_nodes.size(0))  # test 节点被重复 train_nodes.size(0) 次
    pairs = torch.stack((rows, cols), dim=-1)  # 每行表示一条候选边 (row, col)

    # 筛选掉不符合条件的边（已存在的边和自环）
    mask = (pairs[:, 0] != pairs[:, 1]) & (adj[pairs[:, 0], pairs[:, 1]] == 0)
    candidate_edges = pairs[mask]

    # 构建候选边的邻接矩阵
    candidate_adj_mask = torch.zeros_like(adj, dtype=torch.bool)
    candidate_adj_mask[candidate_edges[:, 0], candidate_edges[:, 1]] = True
    candidate_adj_mask = candidate_adj_mask | candidate_adj_mask.T
    print("候选边生成结束")
    return candidate_adj_mask





def compute_beta( len_train, len_unlabeled):
    # 训练集和训练集之间的边数
    train_train = len_train*(len_train-1) / 2

    # 测试集和测试集之间的边数
    test_test = len_unlabeled*(len_unlabeled-1) / 2

    # 训练集和测试集之间的边数
    # 直接从 adj 矩阵中获取训练集和测试集之间的边数
    train_test = len_train*len_unlabeled

    # 返回lambda_1, lambda_2, lambda_3的比例
    return train_train / (train_train + train_test + test_test), train_test / (train_train + train_test + test_test), test_test / (train_train + train_test + test_test)

def compute_w( beta_TT, beta_Tt, beta_tt,epsilon = 0.001):
    WTT = 1 / math.pow(math.exp(beta_TT - 0.5) + epsilon,2)
    WTt = 1 /  math.pow((math.exp(beta_Tt - 0.5) + epsilon),2)
    Wtt = 1 /  math.pow((math.exp(beta_tt - 0.5) + epsilon),2)
    W = WTT +WTt +Wtt
    return WTT/W,WTt/W,Wtt/W

def classify_edge(row, column, idx_train, idx_val, idx_test):
    """
    判断一条边的类型：训练边、验证边、测试边或混合边。
    """
    if row in idx_train and column in idx_train:
        return "train_train"
    elif (row in idx_train and column in idx_val) or (row in idx_val and column in idx_train):
        return "train_val"
    elif (row in idx_train and column in idx_test) or (row in idx_test and column in idx_train):
        return "train_test"
    elif row in idx_test and column in idx_test:
        return "test_test"
    elif (row in idx_test and column in idx_val) or (row in idx_val and column in idx_test):
        return "test_val"
    elif row in idx_val and column in idx_val:
        return "val_val"
    else:
        return "others"


# def visualize(h, data, name):
#
