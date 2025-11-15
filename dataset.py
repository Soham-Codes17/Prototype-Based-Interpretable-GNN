import torch 
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
from dig.xgraph.dataset import MoleculeDataset, SynGraphDataset, SentiGraphDataset, BA_LRP
from torch_geometric.datasets import Planetoid
import numpy as np

from torch_geometric.data import Dataset, download_url

import scipy.sparse as sp
import os.path as osp
import os
import urllib.request
import sys
import pickle as pkl
import networkx as nx
from sklearn.model_selection import train_test_split


def idx_to_mask(indices, n):
    mask = torch.zeros(n, dtype=torch.bool)
    mask[indices] = True
    return mask


def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    """Split nodes into train/val/test following Nettack/Mettack settings."""
    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(
        idx,
        random_state=None,
        train_size=train_size + val_size,
        test_size=test_size,
        stratify=stratify
    )

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(
        idx_train_and_val,
        random_state=None,
        train_size=(train_size / (train_size + val_size)),
        test_size=(val_size / (train_size + val_size)),
        stratify=stratify
    )

    return idx_train, idx_val, idx_test


class Dataset():
    """Dataset class for classic citation network datasets."""

    def __init__(self, root, name, seed=None):
        self.name = name.lower()
        assert self.name in ['cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed'], \
            'Currently only support cora, citeseer, cora_ml, polblogs, pubmed'

        self.seed = seed
        self.url = f'https://raw.githubusercontent.com/danielzuegner/gnn-meta-attack/master/data/{self.name}.npz'
        self.root = osp.expanduser(osp.normpath(root))
        self.data_folder = osp.join(root, self.name)
        self.data_filename = self.data_folder + '.npz'

        self.adj, self.features, self.labels = self.load_data()
        self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test()

    def get_train_val_test(self):
        return get_train_val_test(
            nnodes=self.adj.shape[0],
            val_size=0.1,
            test_size=0.8,
            stratify=self.labels,
            seed=self.seed
        )

    def load_data(self):
        print(f'Loading {self.name} dataset...')
        if self.name == 'pubmed':
            return self.load_pubmed()

        if not osp.exists(self.data_filename):
            self.download_npz()

        adj, features, labels = self.get_adj()
        return adj, features, labels

    def download_npz(self):
        print(f'Downloading from {self.url} to {self.data_filename}')
        try:
            urllib.request.urlretrieve(self.url, self.data_filename)
        except Exception as e:
            raise Exception(f"Download failed: {e}")

    def download_pubmed(self, name):
        url = 'https://raw.githubusercontent.com/tkipf/gcn/master/gcn/data/'
        urllib.request.urlretrieve(url + name, osp.join(self.root, name))

    def load_pubmed(self):
        dataset = 'pubmed'
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            name = f"ind.{dataset}.{names[i]}"
            data_filename = osp.join(self.root, name)

            if not osp.exists(data_filename):
                self.download_pubmed(name)

            with open(data_filename, 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_file = f"ind.{dataset}.test.index"
        if not osp.exists(osp.join(self.root, test_idx_file)):
            self.download_pubmed(test_idx_file)

        with open(osp.join(self.root, test_idx_file)) as f:
            test_idx_reorder = list(map(int, f.readlines()))
        test_idx_range = np.sort(test_idx_reorder)

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = np.where(labels)[1]
        return adj, features, labels

    def get_adj(self):
        adj, features, labels = self.load_npz(self.data_filename)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1
        lcc = self.largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        labels = labels[lcc]
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()
        return adj, features, labels

    def load_npz(self, file_name, is_sparse=True):
        with np.load(file_name) as loader:
            if is_sparse:
                adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                     loader['adj_indptr']), shape=loader['adj_shape'])
                if 'attr_data' in loader:
                    features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                              loader['attr_indptr']), shape=loader['attr_shape'])
                else:
                    features = None
                labels = loader.get('labels')
            else:
                adj = loader['adj_data']
                features = loader.get('attr_data', np.eye(adj.shape[0]))
                labels = loader.get('labels')

        if features is None:
            features = np.eye(adj.shape[0])
        features = sp.csr_matrix(features, dtype=np.float32)
        return adj, features, labels

    def largest_connected_components(self, adj, n_components=1):
        _, component_indices = sp.csgraph.connected_components(adj)
        component_sizes = np.bincount(component_indices)
        components_to_keep = np.argsort(component_sizes)[::-1][:n_components]
        nodes_to_keep = [
            idx for idx, comp in enumerate(component_indices)
            if comp in components_to_keep
        ]
        print(f"Selecting {n_components} largest connected components")
        return nodes_to_keep


class MyCiteseer():
    def __init__(self, data):
        self.data = data
        self.num_node_features = data.x.shape[1]
        self.num_classes = data.y.max().item() + 1

    def __len__(self):
        return 1

    def get(self, idx):
        return self.data

    def __getitem__(self, idx):
        return self.data


def get_dataset(dataset_root, dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name in list(MoleculeDataset.names.keys()):
        return MoleculeDataset(root=dataset_root, name=dataset_name)
    elif dataset_name in ['graph_sst2', 'graph_sst5', 'twitter']:
        return SentiGraphDataset(root=dataset_root, name=dataset_name)
    elif dataset_name in list(SynGraphDataset.names.keys()):
        return SynGraphDataset(root=dataset_root, name=dataset_name)
    elif dataset_name == 'ba_lrp':
        return BA_LRP(root=dataset_root)
    elif dataset_name == 'citeseer':
        np.random.seed(15)
        data = Dataset(root=dataset_root, name=dataset_name)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        from torch_geometric.utils import from_scipy_sparse_matrix
        from torch_geometric.data import Data
        edge_index, _ = from_scipy_sparse_matrix(adj)
        features = torch.FloatTensor(features.toarray())
        labels = torch.LongTensor(labels)
        train_mask = idx_to_mask(idx_train, features.shape[0])
        test_mask = idx_to_mask(idx_test, features.shape[0])
        val_mask = idx_to_mask(idx_val, features.shape[0])
        data = Data(x=features, y=labels, edge_index=edge_index,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        return MyCiteseer(data)
    elif dataset_name == 'cora':
        return Planetoid(root=dataset_root, name="Cora", split="public")
    elif dataset_name == 'pubmed':
        return Planetoid(root=dataset_root, name="PubMed", split="public")
    elif dataset_name == 'mutag':
        # ✅ Added MUTAG support for your ProtoP-GNN project
        return MoleculeDataset(root=dataset_root, name='mutag')
    else:
        raise ValueError(f"{dataset_name} is not defined.")


def get_dataloader(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=0):
    """Return training, validation, and testing dataloaders."""
    g = torch.Generator()
    g.manual_seed(seed)

    if not random_split_flag and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "split idx missing"
        split_indices = dataset.supplement['split_indices']
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()
        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval
        print(num_train, num_eval, num_test)
        train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test], generator=g)

    def _init_fn(worker_id):
        np.random.seed(seed)
        torch.manual_seed(seed)

    dataloader = {
        'train': DataLoader(train, batch_size=batch_size, shuffle=True, worker_init_fn=_init_fn),
        'eval': DataLoader(eval, batch_size=batch_size, shuffle=False, worker_init_fn=_init_fn),
        'test': DataLoader(test, batch_size=batch_size, shuffle=False, worker_init_fn=_init_fn)
    }
    return dataloader
