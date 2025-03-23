import os.path as osp

from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon, TUDataset, PolBlogs
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset

def get_dataset(path, name):
    # 确保传入的数据集名称是支持的范围
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS',
                    'Coauthor-CS', 'Coauthor-Phy', 'Amazon-Computers', 'Amazon-Photo',
                    'ogbn-arxiv', 'ogbg-code', 'Proteins', 'PolBlogs']

    # 如果数据集名称是 DBLP，将其转换为小写形式 'dblp'
    name = 'dblp' if name == 'DBLP' else name

    # 如果选择的是 Proteins 数据集，则加载 TUDataset 数据集
    if name == 'Proteins':
        return TUDataset(root=path, name='PROTEINS', transform=T.NormalizeFeatures())

    # 如果选择的是 Coauthor-CS 数据集
    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    # 如果选择的是 Coauthor-Phy 数据集
    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    # 如果选择的是 WikiCS 数据集
    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    # 如果选择的是 Amazon-Computers 数据集
    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    # 如果选择的是 Amazon-Photo 数据集
    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    # 如果选择的是 OGB (Open Graph Benchmark) 数据集
    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(path, 'OGB'), name=name, transform=T.NormalizeFeatures())
    if name == 'PolBlogs':
        return PolBlogs(root=path, transform=T.NormalizeFeatures())

    # 对于 Citation 数据集（如 dblp 或其他 Planetoid 数据集），进行处理
    if name == 'dblp':
        dataset_class = CitationFull  # 使用 CitationFull 数据集类
    else:
        dataset_class = Planetoid  # 使用 Planetoid 数据集类

    return dataset_class(osp.join(path, 'Citation'), name, transform=T.NormalizeFeatures())
