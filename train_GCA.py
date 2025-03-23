import argparse
import os
import os.path as osp
import random
import shutil

import nni
import time
import pickle as pkl

import numpy as np
import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected

from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted, feature_drop_weights_dense, drop_edge_weighted_train, \
    drop_feature_weighted_train
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset

# 训练函数
def train():
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清空梯度

    # 随机丢弃边
    def drop_edge(idx: int):
        global drop_weights

        if param['drop_scheme'] == 'uniform':  # 如果丢弃方案是uniform（均匀丢弃）
            return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:  # 如果丢弃方案是基于度、特征或PageRank
            if args.defence:
                return drop_edge_weighted_train(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], train_mask=data.train_mask, test_mask=data.test_mask,threshold=0.7)
            else:
                return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

    # 丢弃边和特征
    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)
    x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
    x_2 = drop_feature(data.x, param['drop_feature_rate_2'])


    # 如果丢弃方案是加权的，重新丢弃特征
    if param['drop_scheme'] in ['pr', 'degree', 'evc']:
        # x_1 = drop_feature_weighted_train(data.x, feature_weights, param['drop_feature_rate_1'],mask=data.test_mask)
        # x_2 = drop_feature_weighted_train(data.x, feature_weights, param['drop_feature_rate_2'],mask=data.test_mask)
        x_1 = drop_feature_weighted(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted(data.x, feature_weights, param['drop_feature_rate_2'])


    # 前向传播
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    # 计算损失并反向传播
    loss = model.loss(z1, z2, batch_size=None)
    loss.backward()
    optimizer.step()  # 更新参数

    return loss.item()  # 返回损失值

# 测试函数
def test(final=False):
    model.eval()  # 设置模型为评估模式
    z = model(data.x, data.edge_index)  # 前向传播得到表示向量
    evaluator = MulticlassEvaluator()  # 多分类评估器
    # 根据数据集选择不同的测试方式
    if args.dataset == 'Cora':
        acc = log_regression(z, data, evaluator, split='cora', num_epochs=3000)['acc']
    elif args.dataset == 'CiteSeer':
        acc = log_regression(z, data, evaluator, split='citeseer', num_epochs=3000)['acc']
    elif args.dataset == 'PolBlogs':
        acc = log_regression(z, data, evaluator, split='PolBlogs', num_epochs=3000)['acc']
    else:
        raise ValueError('Please check the split first!')  # 如果数据集不符合，抛出异常

    # 如果使用NNI，报告中间或最终结果
    if final and use_nni:
        nni.report_final_result(acc)
    elif use_nni:
        nni.report_intermediate_result(acc)

    return acc  # 返回准确率

# 主程序
if __name__ == '__main__':
    shutil.rmtree("dataset/Cora/Citation/Cora/processed", ignore_errors=True)
    shutil.rmtree("dataset/CiteSeer/Citation/CiteSeer/processed", ignore_errors=True)
    shutil.rmtree("dataset/PolBlogs/processed", ignore_errors=True)
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')  # 设备（GPU或CPU）
    parser.add_argument('--dataset', type=str, default='Cora')  # 数据集
    parser.add_argument('--param', type=str, default='local:general.json')  # 配置文件
    parser.add_argument('--seed', type=int, default=39788)  # 随机种子
    parser.add_argument('--verbose', type=str, default='train,eval,final')  # 日志输出
    parser.add_argument('--save_split', action="store_true")  # 是否保存数据集划分
    parser.add_argument('--load_split', action="store_true")  # 是否加载数据集划分
    parser.add_argument('--attack_method', type=str, default='PPP')  # 攻击方法    Ada-GCLA
    parser.add_argument('--perturb', action='store_true', default=True)  # 是否使用攻击扰动
    parser.add_argument('--defence', type=bool, default=False)  # 是否使用攻击扰动
    parser.add_argument('--attack_rate', type=float, default=0.10)  # 攻击比例
    parser.add_argument('--train_set_rate', type=float, default=0.10)  # 数据集划分训练集比例
    parser.add_argument('--val_set_rate', type=float, default=0.10)  # 数据集划分验证集比例

    # 解析参数
    args = parser.parse_args()

    # 读取配置文件中的参数
    sp = SimpleParam()
    param = sp(source=args.param, preprocess='nni')

    # 合并命令行参数和配置文件参数
    for key in vars(args).keys():
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    use_nni = True  # 启用NNI进行超参数调优
    np.random.seed(12345)  # Numpy的随机种子
    torch.manual_seed(args.seed)  # Pytorch的CPU随机种子
    torch.cuda.manual_seed(args.seed)  # Pytorch的GPU随机种子（如果有GPU）

    device = torch.device(args.device)  # 设置设备（GPU或CPU）

    # 加载数据集
    path = osp.expanduser('dataset')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    train_rate = args.train_set_rate
    val_rate = args.val_set_rate
    split = generate_split(dataset[0].num_nodes, train_ratio=train_rate, val_ratio=val_rate)
    dataset.data.train_mask = split[0]
    dataset.data.test_mask = split[1]
    dataset.data.val_mask = split[2]
    if args.dataset == 'PolBlogs':

        data = dataset.data
        data.x = torch.randn(data.num_nodes, 32)  # 随机初始化特征
        dataset.data.x = torch.randn(data.num_nodes, 32)  # 随机初始化特征
    else:
        data = dataset[0]  # 获取数据集的第一个元素

    # 加载中毒邻接矩阵
    if args.perturb:
        try:
            perturbed_adj = pkl.load(open('poisoned_adj/%s_%s_%f_%.2f_adj.pkl' % (args.dataset, args.attack_method, args.attack_rate, train_rate), 'rb')).to(device)
        except:
            perturbed_adj = torch.load('poisoned_adj/%s_%s_%f_%.2f_adj.pkl' % (args.dataset, args.attack_method, args.attack_rate, train_rate), map_location=device)
        data.edge_index = perturbed_adj.nonzero().T  # 更新扰动后的邻接矩阵
        print("load poisoned_adj: poisoned_adj/%s_%s_%f_%.2f_adj.pkl" % (args.dataset, args.attack_method, args.attack_rate,train_rate ))
    else:
        print("未添加中毒攻击")  # 如果没有攻击扰动，输出提示

    if args.defence:
        print("添加防御")
    else:
        print("不添加防御")
    data = data.to(device)  # 将数据移到设备上

    idx_train = torch.where(data.train_mask)[0]
    idx_val = torch.where(data.val_mask)[0]
    idx_test = torch.where(data.test_mask)[0]
    idx_unlabeled = torch.cat((idx_val, idx_test)).unique()

    # 设置模型和优化器
    encoder = Encoder(data.num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    # 根据丢弃策略选择权重
    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif param['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif param['drop_scheme'] == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    # 为特征丢弃选择合适的权重
    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1], num_nodes=data.num_nodes)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif param['drop_scheme'] == 'pr':
        node_pr = compute_pr(data.edge_index)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif param['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)

    log = args.verbose.split(',')
    print('Begin training....')

    start_time = time.time()  # 记录开始时间
    for epoch in range(1, 3000 + 1):
        loss = train()
        if epoch % 100 == 0:
            end_time = time.time()  # 记录每 100 轮后的时间
            training_time = end_time - start_time  # 计算从上一次输出到现在的时间
            start_time = end_time  # 更新开始时间为当前时间
            print(f'(T) | Epoch={epoch:03d}, loss={loss:f}, training time={training_time:.2f}s')
    sum = 0.0
    for epoch in range(0, 20):
        acc = test()
        sum += acc
    print("Average ACC: %.4f" % (sum / 20))



    print(
        f"dataset:{args.dataset}   attack_method:{args.attack_method}   attack_rate:{args.attack_rate * 100}%   run_device:{device}  num_epoch:{param['num_epochs']}   learning_rate:{param['learning_rate']}")
    print(idx_train)
    # print(idx_val)