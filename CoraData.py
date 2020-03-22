#!/usr/bin/env python
# coding: utf-8

from collections import namedtuple
import os
import os.path as osp
import urllib.request
import numpy as np
import pickle
import ssl
import itertools
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.init as init

Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])

class CoraData(object):
    download_url = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root = "cora", rebuild = False):
        self.data_root = data_root
        save_file = osp.join(self.data_root, "processed_cora.pkl")
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self.maybe_download()
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        return self._data

    def process_data(self):
        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(
            osp.join(self.data_root, "raw", name)) for name in self.filenames]
        train_index = np.arange(y.shape[0]) #1000*1
        val_index = np.arange(y.shape[0], y.shape[0] + 500) #500*1
        sorted_test_index = sorted(test_index) #1000*1

        x= np.concatenate((allx, tx), axis=0) #2708*1433
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1) #2708*1

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        adjacency = self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x, y=y, adjacency=adjacency,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    def maybe_download(self):
        save_path = osp.join(self.data_root, "raw")
        for name in self.filenames:
            if not osp.exists(osp.join(save_path, name)):
                self.download_data("{}/{}".format(self.download_url, name), save_path)

    # 创建邻接矩阵
    @staticmethod
    def build_adjacency(adj_dict):
        edge_index = []
        num_nodes = len(adj_dict)
        for scr, dst in adj_dict.items():
            edge_index.extend([scr, v] for v in dst)
            edge_index.extend([v, scr] for v in dst)
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                                   (edge_index[:,0], edge_index[:,1])),
                                  shape=(num_nodes, num_nodes),
                                  dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

    @staticmethod
    def download_data(url, save_path):
        if not osp.exists(save_path):
            os.makedirs(save_path)
        print(url)
        ssl._create_default_https_context = ssl._create_unverified_context
        data = urllib.request.urlopen(url)
        filename = osp.split(url)[-1]

        with open(osp.join(save_path, filename), 'wb') as f:
            f.write(data.read())
        return True

# 模型训练部分


# 图卷基层定义
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, out_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = out_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, out_dim))
        if self.use_bias:
            self.bias = nn.parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 参数初始化
    def reset_parameters(self):
        init.kaiming_normal_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)


# 模型定义
class GcnNet(nn.Module):
    def __init__(self, input_dim=1433):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 16)
        self.gcn2 = GraphConvolution(16, 7)

# 定义超参数
learning_rate = 0.1
weight_decay = 5e-4
epochs = 200


#
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: {}".format(device))

x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
y.requires_grad_(True)
print(y)
z = y * y * 3
print(z.grad_fn)
out = z.mean()
out.backward()
print(z, out)
print("x's grad is {}".format(x.grad))
print("z's grad is {}".format(y.grad))

print("Done!!!")