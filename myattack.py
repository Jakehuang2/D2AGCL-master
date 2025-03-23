import os
import shutil
import warnings

import numpy as np

from CLGAqifa import compute_lambda

warnings.filterwarnings("ignore")
from torch.nn.modules.module import Module
import argparse
import os.path as osp
import time
import pickle as pkl
import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected, dense_to_sparse, is_undirected, to_networkx, contains_isolated_nodes
from simple_param.sp import SimpleParam
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted, feature_drop_weights_dense
from pGRACE.utils import get_activation, compute_pr, eigenvector_centrality, generate_split, getCandidateEdge, \
    show_perturbed_edge, compute_beta, getCandidateEdge2, classify_edge, generate_split_sequential, compute_w
from pGRACE.dataset import get_dataset
from differentiable_models.gcn import GCN
from differentiable_models.model import GRACE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Metacl(Module):
    def __init__(self, args, dataset, param, device):
        super(Metacl, self).__init__()
        self.model = None
        self.optimizer = None
        self.param = param
        self.args = args
        self.device = device
        self.dataset = dataset
        self.data = dataset.data.to(device)
        self.drop_weights = None
        self.feature_weights = None

    def drop_edge(self, p):
        if self.param['drop_scheme'] == 'uniform':
            return dropout_adj(self.data.edge_index, p=p)[0]
        elif self.param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(self.data.edge_index, self.drop_weights, p=p,
                                      threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

    def train_gcn(self):
        self.model.train()
        self.optimizer.zero_grad()
        edge_index_1 = self.drop_edge(self.param['drop_edge_rate_1'])
        edge_index_2 = self.drop_edge(self.param['drop_edge_rate_2'])
        x_1 = drop_feature(self.data.x, self.param['drop_feature_rate_1'])
        x_2 = drop_feature(self.data.x, self.param['drop_feature_rate_2'])
        edge_sp_adj_1 = torch.sparse.FloatTensor(edge_index_1,
                                                 torch.ones(edge_index_1.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        edge_sp_adj_2 = torch.sparse.FloatTensor(edge_index_2,
                                                 torch.ones(edge_index_2.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        if self.param['drop_scheme'] in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted(self.data.x, self.feature_weights, self.param['drop_feature_rate_1'])
            x_2 = drop_feature_weighted(self.data.x, self.feature_weights, self.param['drop_feature_rate_2'])
        z1 = self.model(x_1, edge_sp_adj_1, sparse=True)
        z2 = self.model(x_2, edge_sp_adj_2, sparse=True)
        loss = self.model.loss(z1, z2, batch_size=None)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_drop_weights(self):
        if self.param['drop_scheme'] == 'degree':
            self.drop_weights = degree_drop_weights(self.data.edge_index).to(self.device)
        elif self.param['drop_scheme'] == 'pr':
            self.drop_weights = pr_drop_weights(self.data.edge_index, aggr='sink', k=200).to(self.device)
        elif self.param['drop_scheme'] == 'evc':
            self.drop_weights = evc_drop_weights(self.data).to(self.device)
        else:
            self.drop_weights = None

        if self.param['drop_scheme'] == 'degree':
            #转成无向图
            edge_index_ = to_undirected(self.data.edge_index)
            #计算无向图的度
            node_deg = degree(edge_index_[1])
            if self.args.dataset == 'WikiCS':
                self.feature_weights = feature_drop_weights_dense(self.data.x, node_c=node_deg).to(self.device)
            else:
                self.feature_weights = feature_drop_weights(self.data.x, node_c=node_deg).to(self.device)
        elif self.param['drop_scheme'] == 'pr':
            node_pr = compute_pr(self.data.edge_index)
            if self.args.dataset == 'WikiCS':
                self.feature_weights = feature_drop_weights_dense(self.data.x, node_c=node_pr).to(self.device)
            else:
                self.feature_weights = feature_drop_weights(self.data.x, node_c=node_pr).to(self.device)
        elif self.param['drop_scheme'] == 'evc':
            node_evc = eigenvector_centrality(self.data)
            if self.args.dataset == 'WikiCS':
                self.feature_weights = feature_drop_weights_dense(self.data.x, node_c=node_evc).to(self.device)
            else:
                self.feature_weights = feature_drop_weights(self.data.x, node_c=node_evc).to(self.device)
        else:
            self.feature_weights = torch.ones((self.data.x.size(1),)).to(self.device)

    def inner_train(self):
        #此处的GCN是代理模型，在生成中毒图的时候用到
        if args.dataset == 'PolBlogs':
            encoder = GCN(self.dataset.data.num_features, self.param['num_hidden'], get_activation(self.param['activation']))
        else:
            encoder = GCN(self.dataset.num_features, self.param['num_hidden'], get_activation(self.param['activation']))
        self.model = GRACE(encoder, self.param['num_hidden'], self.param['num_proj_hidden'], self.param['tau']).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.param['learning_rate'],
            weight_decay=self.param['weight_decay']
        )
        self.compute_drop_weights()
        for epoch in range(1, self.param['num_epochs'] + 1):
            loss = self.train_gcn()

    def compute_gradient(self, pe1, pe2, pf1, pf2):
        self.model.eval()
        self.compute_drop_weights()
        edge_index_1 = self.drop_edge(pe1)
        edge_index_2 = self.drop_edge(pe2)
        x_1 = drop_feature(self.data.x, pf1)
        x_2 = drop_feature(self.data.x, pf2)
        edge_sp_adj_1 = torch.sparse.FloatTensor(edge_index_1,
                                                 torch.ones(edge_index_1.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        edge_sp_adj_2 = torch.sparse.FloatTensor(edge_index_2,
                                                 torch.ones(edge_index_2.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        if self.param['drop_scheme'] in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted(self.data.x, self.feature_weights, pf1)
            x_2 = drop_feature_weighted(self.data.x, self.feature_weights, pf2)
        edge_adj_1 = edge_sp_adj_1.to_dense()
        edge_adj_2 = edge_sp_adj_2.to_dense()
        edge_adj_1.requires_grad = True
        edge_adj_2.requires_grad = True
        z1 = self.model(x_1, edge_adj_1, sparse=False)
        z2 = self.model(x_2, edge_adj_2, sparse=False)
        loss = self.model.loss(z1, z2, batch_size=None)
        loss.backward()
        return edge_adj_1.grad, edge_adj_2.grad

    def add_perturbed(self,candidate_adj_mask_1d):
        start = time.time()
        self.inner_train()
        adj_1_grad, adj_2_grad = self.compute_gradient(self.param['drop_edge_rate_1'],
                                                       self.param['drop_edge_rate_2'],
                                                       self.param['drop_feature_rate_1'],
                                                       self.param['drop_feature_rate_2'])
        grad_sum = adj_1_grad + adj_2_grad
        grad_sum_1d = grad_sum.view(-1)
        filtered_grad_sum_1d = grad_sum_1d.clone()

        filtered_grad_sum_1d[~candidate_adj_mask_1d] = 0
        filtered_grad_sum_1d_abs =torch.abs(filtered_grad_sum_1d)
        values, indices = filtered_grad_sum_1d_abs.sort(descending=True)


        i = -1
        # flag = 0
        while True:
            i += 1
            index = int(indices[i])
            row = int(index / self.data.num_nodes)
            column = index % self.data.num_nodes
            grad_value = filtered_grad_sum_1d[index]
            if [row, column] in perturbed_edges or  [column, row] in perturbed_edges:
                continue
            # if labels[row].item() == labels[column].item():
            #     continue
            # 增加边，考虑在内
            if grad_value>0 and adj[row, column] == 0:
                adj[row, column] = 1
                adj[column, row] = 1
                perturbed_edges.append([row, column])
                perturbed_edges.append([column, row])
                break
                # flag+=1
                # if flag>=num:
                #     break
                # else:
                #     continue
            elif grad_value < 0 and adj[row, column] == 1:
                print("删除边了")
                adj[row, column] = 0
                adj[column, row] = 0
                perturbed_edges.append([row, column])
                perturbed_edges.append([column, row])
                break
                # flag += 1
                # if flag >= num:
                #     break
                # else:
                #     continue

        self.data.edge_index = dense_to_sparse(adj)[0]
        end = time.time()
        current_edge = len(perturbed_edges) / 2

        percentage_completed = (current_edge / n_perturbations) * 100
        print(
            'Perturbing edges: %d/%d. Finished in %.2fs. %.2f%% complete. Estimated time remaining: %.2fs %s  %d' % (
                current_edge, n_perturbations, end - start, percentage_completed,
                (n_perturbations - current_edge) * (end - start),classify_edge(row,column,idx_train,idx_val,idx_test),i
            )
        )
        print(labels[row].item(),"--", labels[column].item(), end="")
        print("True" if labels[row].item() == labels[column].item() else "False")





    def attack(self):
        print('Begin perturbing.....')
        candidate_train_mask_1d = getCandidateEdge(data, adj, idx_train, idx_train).view(-1)
        candidate_train_test_mask_1d = getCandidateEdge(data, adj, idx_train, idx_unlabeled).view(-1)
        candidate_test_mask_1d = getCandidateEdge(data, adj, idx_unlabeled, idx_unlabeled).view(-1)
        cnt_train = 0
        cnt_train_test = 0
        cnt_test = 0

        while cnt_train < n_train or cnt_train_test < n_train_test or cnt_test < n_test:
            if cnt_train < n_train:
                self.add_perturbed(candidate_train_mask_1d)
                cnt_train += 1

            if cnt_train_test < n_train_test:
                self.add_perturbed(candidate_train_test_mask_1d)
                cnt_train_test += 1

            if cnt_test < n_test:
                self.add_perturbed(candidate_test_mask_1d)
                cnt_test += 1

        print('Number of perturbed edges: %d' % (len(perturbed_edges)/2))
        output_adj = adj.to(device)

        # 新增：获取节点嵌入
        edge_index = dense_to_sparse(output_adj)[0]
        edge_sp_adj = torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.shape[1]).to(device), [data.num_nodes, data.num_nodes])
        z = self.model.encoder(data.x, edge_sp_adj, sparse=True).detach().cpu().numpy()

        # 新增：t-SNE 降维
        tsne = TSNE(n_components=2, random_state=39788)
        z_2d = tsne.fit_transform(z)

        # 新增：可视化
        plt.figure(figsize=(10, 8))
        plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels.cpu().numpy(), cmap='rainbow', s=10)
        plt.title('t-SNE Visualization of Poisoned Graph')
        plt.colorbar()
        plt.savefig('poisoned_adj/%s_PPP_%f_%.2f_tsne.png' % (args.dataset, attack_rate, train_rate))
        plt.show()

        return output_adj


if __name__ == '__main__':
    shutil.rmtree("dataset/Cora/Citation/Cora/processed", ignore_errors=True)
    shutil.rmtree("dataset/CiteSeer/Citation/CiteSeer/processed", ignore_errors=True)
    shutil.rmtree("dataset/PolBlogs/processed", ignore_errors=True)
    os.makedirs('poisoned_adj', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--param', type=str, default='local:general.json')
    parser.add_argument('--attack_rate', type=float, default=0.10)  # 攻击比例


    parser.add_argument('--train_set_rate', type=float, default=0.11)  # 数据集划分训练集比例
    parser.add_argument('--val_set_rate', type=float, default=0.10)  # 数据集划分验证集比例
    # parser.add_argument('--unionattack', type=int, default=1)  # 数据集划分验证集比例
    # add hyper-parameters into parser
    args = parser.parse_args()
    attack_rate = args.attack_rate
    print(attack_rate)
    train_rate = args.train_set_rate
    val_rate = args.val_set_rate
    # attack_num = args.unionattack
    # parse param
    sp = SimpleParam()
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in vars(args).keys():
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    #
    np.random.seed(12345)  # Numpy的随机种子
    torch.manual_seed(args.seed)  # Pytorch的CPU随机种子
    torch.cuda.manual_seed(args.seed)  # Pytorch的GPU随机种子（如果有GPU）


    device = torch.device(args.device)

    path = osp.expanduser('dataset')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    split = generate_split(dataset[0].num_nodes, train_ratio=train_rate, val_ratio=val_rate)
    dataset.data.train_mask = split[0]
    dataset.data.test_mask = split[1]
    dataset.data.val_mask = split[2]

    if args.dataset == 'PolBlogs':
        split = generate_split(dataset[0].num_nodes, train_ratio=train_rate, val_ratio=val_rate)
        dataset.data.train_mask = split[0]
        dataset.data.test_mask = split[1]
        dataset.data.val_mask = split[2]
        data = dataset.data
        data.x = torch.randn(data.num_nodes, 32)
        dataset.data.x = torch.randn(data.num_nodes, 32)
    else:
        data = dataset[0]
    data.to(device)
    idx_train = torch.where(data.train_mask)[0]
    idx_val = torch.where(data.val_mask)[0]
    idx_test = torch.where(data.test_mask)[0]
    idx_unlabeled = torch.cat((idx_val, idx_test)).unique()
    labels = data.y

    adj_sp = torch.sparse.FloatTensor(data.edge_index, torch.ones(data.edge_index.shape[1]).to(device),
                                      [data.num_nodes, data.num_nodes])
    adj = adj_sp.to_dense()
    # 根据数据集划分情况动态攻击
    beta_TT, beta_Tt, beta_tt = compute_beta(len(idx_train), len(idx_unlabeled))
    w_TT, w_Tt, w_tt = compute_w(beta_TT,beta_Tt,beta_tt)
    n_perturbations = int(adj.sum().item() * attack_rate / 2)

    n_train = int(n_perturbations * w_TT)
    n_test = int(n_perturbations * w_tt)
    n_train_test = n_perturbations-n_train-n_test
    perturbed_edges=[]
    print(f"扰动边数量：train_train：{n_train}, train_test:{n_train_test}, test_test:{n_test}")
    model = Metacl(args, dataset, param, device).to(device)
    poisoned_adj = model.attack()
    pkl.dump(poisoned_adj.to(torch.device('cpu')), open('poisoned_adj/%s_PPP_%f_%.2f_adj.pkl' % (args.dataset,attack_rate,train_rate), 'wb'))
    print('---10% perturbed adjacency matrix saved---')
    show_perturbed_edge(data,perturbed_edges)

    # print(idx_train)
    # print(idx_val)
