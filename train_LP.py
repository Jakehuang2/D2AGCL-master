import numpy as np
import argparse
import os.path as osp
import nni
import time
import pickle as pkl
import torch
from torch_geometric.utils import degree, to_undirected
from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE
from pGRACE.functional import drop_edge_weighted, \
    degree_drop_weights, feature_drop_weights, feature_drop_weights_dense
from pGRACE.eval import link_prediction, LPEvaluator
from pGRACE.utils import get_base_model, get_activation, generate_split
from pGRACE.dataset import get_dataset
def train():
    model.train()
    optimizer.zero_grad()
    def drop_edge(idx: int):
        global drop_weights
        if param['drop_scheme'] in ['uniform', 'degree', 'evc', 'pr']:
            return drop_edge_weighted(train_edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')
    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)
    # edge_sp_adj_1 = torch.sparse_coo_tensor(
    #     edge_index_1,  # 边索引，形状为 (2, num_edges)
    #     torch.ones(edge_index_1.shape[1]).to(device),  # 边的权重，形状为 (num_edges,)
    #     torch.Size([data.num_nodes, data.num_nodes])  # 矩阵的形状 (num_nodes, num_nodes)
    # ).to(device)
    # edge_sp_adj_2 = torch.sparse_coo_tensor(
    #     edge_index_2,  # 边索引，形状为 (2, num_edges)
    #     torch.ones(edge_index_2.shape[1]).to(device),  # 边的权重，形状为 (num_edges,)
    #     torch.Size([data.num_nodes, data.num_nodes])  # 矩阵的形状 (num_nodes, num_nodes)
    # ).to(device)
    # edge_adj_1 = edge_sp_adj_1.to_dense()
    # edge_adj_2 = edge_sp_adj_2.to_dense()
    x_1 = data.x
    x_2 = data.x
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    loss = model.loss(z1, z2, batch_size=1024 if args.dataset == 'Coauthor-Phy' else None)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(final=False):
    model.eval()
    z = model(data.x, train_edge_index)
    evaluator = LPEvaluator()
    auc = link_prediction(z, data.edge_index, train_edge_index, val_edge_index, test_edge_index, data.num_nodes, evaluator,
                          num_epochs=3000)['auc']
    if final and use_nni:
        nni.report_final_result(auc)
    elif use_nni:
        nni.report_intermediate_result(auc)
    return auc
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='CiteSeer')
    parser.add_argument('--param', type=str, default='local:general.json')
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--perturb', action='store_true', default=True)   # 是否使用攻击扰动
    parser.add_argument('--attack_method', type=str, default='CLGAQIFA')
    parser.add_argument('--attack_prop', type=float, default=0.05)
    parser.add_argument('--drop_prop', type=float, default=0.20)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--attack_rate', type=float, default=0.05)

    # add hyper-parameters into parser
    args = parser.parse_args()
    # parse param
    sp = SimpleParam()
    param = sp(source=args.param, preprocess='nni')
    # merge cli arguments and parsed param
    for key in vars(args).keys():
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    device = torch.device(args.device)
    path = osp.expanduser('dataset')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    use_nni = True
    if args.dataset == 'PolBlogs':
        data = dataset.data
        data.x = torch.randn(data.num_nodes, 32)
        dataset.data.x = torch.randn(data.num_nodes, 32)
    else:
        data = dataset[0]
    try:
        perturbed_adj = pkl.load(open('poisoned_adj/%s_%s_%f_%d_adj.pkl' % (args.dataset, args.attack_method, args.attack_rate, param['num_epochs']), 'rb')).to(device)
    except:
        perturbed_adj = torch.load('poisoned_adj/%s_%s_%f_%d_adj.pkl' % (args.dataset, args.attack_method, args.attack_rate, param['num_epochs']), map_location=device)
    data.edge_index = perturbed_adj.nonzero().T
    data = data.to(device)
    # generate edge split
    bidirected_edge_index = data.edge_index.cpu().numpy()
    index = np.where(bidirected_edge_index[0]<bidirected_edge_index[1])[0]
    undirected_edge_index = torch.Tensor(bidirected_edge_index[:, index]).long().to(device)
    train_mask, test_mask, val_mask = generate_split(int(undirected_edge_index.shape[1]), train_ratio=0.7, val_ratio=0.1)
    train_edge_index = to_undirected(undirected_edge_index[:, train_mask])
    test_edge_index = to_undirected(undirected_edge_index[:, test_mask])
    val_edge_index = to_undirected(undirected_edge_index[:, val_mask])
    # assert int(train_edge_index.shape[1]) + int(test_edge_index.shape[1]) + int(val_edge_index.shape[1]) == data.num_edges
    encoder = Encoder(data.num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )
    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(train_edge_index).to(device)
    else:
        drop_weights = torch.ones_like(train_edge_index[0], dtype=torch.float)
    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(train_edge_index)
        node_deg = degree(edge_index_[1], num_nodes=dataset.data.num_nodes)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)
    log = args.verbose.split(',')
    print('Begin training....')
    best_auc = 0
    best_epoch = 0
    for epoch in range(1, 3000 + 1):
        start = time.time()
        loss = train()
        end = time.time()
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, training time={end - start}')
    print("epoch 3000 end")
    auc = test(final=True)
    if 'final' in log:
        print(f'auc={auc}')