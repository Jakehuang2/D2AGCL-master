import torch
from torch_geometric.utils import degree, to_undirected

from pGRACE.utils import compute_pr, eigenvector_centrality


def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def drop_feature_weighted_train(x, w, p: float, threshold: float = 0.7, mask: torch.BoolTensor = None):
    """
    丢弃特征，只针对训练集或测试集的节点进行特征丢弃，并且考虑特征权重。

    Parameters:
    - x: 输入的特征矩阵，形状为 (num_nodes, num_features)
    - w: 特征权重，形状为 (num_features,)
    - p: 丢弃特征的比例
    - threshold: 丢弃概率的阈值
    - mask: 训练集或测试集的掩码，形状为 (num_nodes,)，布尔类型。只有掩码为True的节点会丢弃特征。

    Returns:
    - 修改后的特征矩阵
    """

    # 根据权重调整丢弃概率
    w = w / w.mean() * p  # 规范化和缩放丢弃概率
    w = w.where(w < threshold, torch.ones_like(w) * threshold)  # 应用阈值

    drop_prob = w  # 丢弃概率为权重调整后的概率

    # 计算丢弃掩码（是否丢弃特征），形状为 (num_features,)
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    # 扩展 drop_mask 到整个节点特征矩阵的形状，确保每个节点的特征有相同的丢弃掩码
    drop_mask_expanded = drop_mask.expand(x.shape[0], -1)  # 形状变为 (num_nodes, num_features)

    # 创建一个副本以修改特征矩阵
    x_dropped = x.clone()

    # 仅对 mask 为 True 的节点进行特征丢弃
    # 我们需要对每个节点的特征进行丢弃，这里mask 是布尔类型，它标识出哪些节点要丢弃特征
    x_dropped[mask] = x_dropped[mask] * (~drop_mask_expanded[mask])

    return x_dropped




def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x


def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def feature_drop_weights_dense(x, node_c):
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def drop_edge_weighted_train(edge_index, edge_weights, p: float, train_mask, test_mask, threshold: float = 1.):
    # 1. 规范化边权重
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)

    # 2. 筛选出训练集与训练集之间的边或训练集与测试集之间的边
    edge_src, edge_dst = edge_index
    train_edges = (train_mask[edge_src] & train_mask[edge_dst])  # 训练集节点之间的边
    test_edges = (train_mask[edge_src] & test_mask[edge_dst]) | (
                test_mask[edge_src] & train_mask[edge_dst])  # 训练集与测试集之间的边

    # 3. 创建一个掩码，表示哪些边属于训练集-训练集或训练集-测试集边
    valid_edges_mask = train_edges | test_edges

    # 4. 仅对这些边进行丢弃操作
    edge_weights = edge_weights * valid_edges_mask.to(torch.float)  # 保留有效边的权重，其他边的权重设为0
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    # # 5. 计算丢弃的边的数量
    # total_edges = edge_weights.shape[0]  # 总边数
    # retained_edges = sel_mask.sum().item()  # 保留的边的数量
    # discarded_edges = total_edges - retained_edges  # 丢弃的边的数量
    #
    # # 6. 打印丢弃的边的数量
    # print(f"Total edges: {total_edges}")
    # print(f"Retained edges: {retained_edges}")
    # print(f"Discarded edges: {discarded_edges}")

    # 7. 返回丢弃后的边
    return edge_index[:, sel_mask]


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    # 1. 规范化边权重
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)

    # 2. 计算保留的边
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    # # 3. 计算丢弃的边的数量
    # total_edges = edge_weights.shape[0]  # 总边数
    # retained_edges = sel_mask.sum().item()  # 保留的边的数量
    # discarded_edges = total_edges - retained_edges  # 丢弃的边的数量
    #
    # # 4. 打印丢弃的边的数量
    # print(f"Total edges: {total_edges}")
    # print(f"Retained edges: {retained_edges}")
    # print(f"Discarded edges: {discarded_edges}")

    # # 5. 获取被丢弃边的掩码
    # discarded_mask = ~sel_mask
    #
    # # 6. 使用掩码获取被丢弃边的索引
    # discarded_edges_index = edge_index[:, discarded_mask]
    # print(discarded_edges_index)
    # 5. 返回丢弃后的边
    return edge_index[:, sel_mask]


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)  # 转换为无向图
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights


def evc_drop_weights(data):
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())