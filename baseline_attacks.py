import numpy as np
import argparse
import os.path as osp
import pickle as pkl
from scipy.sparse import csr_matrix

from pGRACE.utils import show_perturbed_edge

import torch
from torch_geometric.utils import contains_isolated_nodes

from pGRACE.dataset import get_dataset

from deeprobust.graph.data import Pyg2Dpr
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack, DICE, Random, MinMax, PGDAttack, NodeEmbeddingAttack
from deeprobust.graph.utils import preprocess


def attack_model(name, adj, features, labels, device):
    if args.rate < 1:
        n_perturbation = int(args.rate * dataset.data.num_edges / 2)
    else:
        n_perturbation = int(args.rate)
    if name == 'metattack':
        model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                          attack_structure=True, attack_features=False, device=device, lambda_=0).to(device)
        perturbed_edges = model.attack(features, adj, labels, idx_train, idx_unlabeled,
                     n_perturbations=n_perturbation, ll_constraint=False)
    elif name == 'dice':
        model = DICE()
        model.attack(adj, labels, n_perturbations=n_perturbation)
    elif name == 'random':
        model = Random()
        model.attack(adj, n_perturbations=n_perturbation)
    elif name == 'minmax':
        adj, features, labels = preprocess(adj, csr_matrix(features), labels, preprocess_adj=False)  # conver to tensor
        model = MinMax(surrogate, nnodes=adj.shape[0], loss_type='CE', device=device).to(device)
        model.attack(features, adj, labels, idx_train, n_perturbations=n_perturbation)
    elif name == 'pgd':
        adj, features, labels = preprocess(adj, csr_matrix(features), labels, preprocess_adj=False) # conver to tensor
        model = PGDAttack(surrogate, nnodes=adj.shape[0], loss_type='CE', device=device).to(device)
        model.attack(features, adj, labels, idx_train, n_perturbations=n_perturbation)
    elif name == 'nodeembeddingattack':
        model = NodeEmbeddingAttack()
        model.attack(adj, attack_type='remove', n_perturbations=n_perturbation)
    else:
        raise ValueError('Invalid name of the attack method!')
    return model,perturbed_edges


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--rate', type=float, default=0.10)
parser.add_argument('--method', type=str, default='metattack')
parser.add_argument('--device', default='cuda:0')

args = parser.parse_args()
device = args.device
path = osp.expanduser('dataset')
path = osp.join(path, args.dataset)
dataset = get_dataset(path, args.dataset)
mapping = None


# else:
#     adj = csr_matrix((np.ones(dataset.data.edge_index.shape[1]),
#                               (dataset.data.edge_index[0], dataset.data.edge_index[1])), shape=(dataset.data.num_nodes, dataset.data.num_nodes))
#     features = dataset.data.x.numpy()
#     labels = dataset.data.y.numpy()

data = Pyg2Dpr(dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)
if args.method in ['metattack', 'minmax', 'pgd']:
    # Setup Surrogate model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
                    with_relu=False, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)
# Setup Attack Model
perturbed_edges=[]
model,perturbed_edges = attack_model(args.method, adj, features, labels, device)
if args.method in ['random', 'dice', 'nodeembeddingattack', 'randomremove', 'randomflip']:
    modified_adj = torch.Tensor(model.modified_adj.todense())
else:
    modified_adj = model.modified_adj  # modified_adj is a torch.tensor
pkl.dump(modified_adj, open('poisoned_adj/%s_%s_%f_adj.pkl' % (args.dataset, args.method, args.rate), 'wb'))
show_perturbed_edge(perturbed_edges)