import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import torch
# from transformers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
import torch_geometric.transforms as T
# from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
import os
import random
# import pandas as pd
import torch
import torch_geometric.transforms as T
from typing import Optional
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import Actor
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.datasets import TUDataset
from sklearn.metrics import r2_score
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_networkx
from torch.nn import Linear
from sklearn.model_selection import KFold
# import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np
import os
import random
# import pandas as pd
import time
# import psutil
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import spspmm
from torch_sparse import coalesce
from torch_sparse import eye
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_scatter import scatter_max
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn import GCNConv, GATConv, LEConv, SAGEConv, GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj, add_self_loops
from typing import Callable, Optional, Union
from torch_sparse import coalesce, transpose
from torch_scatter import scatter
from torch import Tensor
from typing import List, Optional, Tuple, Union
import math
import torch
from torch import Tensor
from torch_geometric.nn.models.mlp import Linear
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn import BatchNorm
import os.path as osp
import time
from math import ceil
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseGCNConv, dense_diff_pool


from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
import random
from torch_geometric.nn import GCNConv
import os.path as osp
import time
from math import ceil
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseGCNConv, dense_diff_pool
max_nodes = 700
data_path = "data"
dataset_dense = TUDataset(
    data_path,
    name="PROTEINS",
    transform=T.Compose([T.ToDense(max_nodes)]),
    use_node_attr=True,
    pre_filter=lambda data: data.num_nodes <= max_nodes,
)
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ASAPooling
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import BatchNorm
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=False, lin=True):
        super().__init__()
        self.conv1 = DenseGCNConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseGCNConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseGCNConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        if lin:
            self.lin = torch.nn.Linear(out_channels, out_channels)
        else:
            self.lin = None
    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x
    def forward(self, x, adj, mask=None):
        x = self.bn(1, self.conv1(x, adj, mask).relu())
        x = self.bn(2, self.conv2(x, adj, mask).relu())
        x = self.bn(3, self.conv3(x, adj, mask).relu())
        if self.lin is not None:
            x = self.lin(x).relu()
        return x
class Net_Diff(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_nodes = 64
        self.gnn1_pool = GNN(dataset_dense.num_features, 64, num_nodes)
        self.gnn1_embed = DenseGCNConv(dataset_dense.num_features, 64)
        num_nodes = 64
        self.gnn2_pool = GNN(64, 64, num_nodes)
        self.gnn2_embed = DenseGCNConv(64, 64)
        self.gnn3_embed = DenseGCNConv(64, 64)
        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, dataset_dense.num_classes)
    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x = F.relu(x)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x = F.relu(x)
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        x = self.gnn3_embed(x, adj)
        x = F.relu(x)
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
model = Net_Diff().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _, _ = model(data.x, data.adj, data.mask)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        total_loss += data.y.size(0) * float(loss)
        optimizer.step()
    return total_loss / len(train_loader.dataset)
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output, _, _ = model(data.x, data.adj, data.mask)
        pred = output.max(dim=1)[1]
        correct += int(pred.eq(data.y.view(-1)).sum())
    return correct / len(loader.dataset)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seeds = [42, 43, 44]
times = []
memories = []
best_val_accs = []
best_test_accs = []
early_stop_patience = 150
tolerance = 0.0001
for seed in seeds:
    set_seed(seed)
    dataset_dense = dataset_dense.shuffle()
    train_ratio = 0.7
    val_ratio = 0.15
    val_ratio = 0.15
    num_total = len(dataset_dense)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)
    num_test = num_total - num_train - num_val
    train_dataset = dataset_dense[:num_train]
    val_dataset = dataset_dense[num_train:num_train + num_val]
    test_dataset = dataset_dense[num_train + num_val:]
    train_loader = DenseDataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_loader = DenseDataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DenseDataLoader(test_dataset, batch_size=512, shuffle=False)
    model = Net_Diff().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    best_val_acc = 0
    epochs_no_improve = 0
    for epoch in range(1, 201):
        loss = train()
        val_acc = test(valid_loader)
        test_acc = test(test_loader)
        if val_acc > best_val_acc + tolerance:
            best_val_acc = val_acc
            best_test_acc = test_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        print(f'Seed: {seed}, Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        if epochs_no_improve >= early_stop_patience:
            print(f'Early stopping at epoch {epoch} for seed {seed}')
            break
    end_time = time.time()
    total_time = end_time - start_time
    memory_allocated = torch.cuda.memory_reserved(device) / (1024 ** 2)  
    times.append(total_time)
    memories.append(memory_allocated)
    best_val_accs.append(best_val_acc)
    best_test_accs.append(best_test_acc)
    torch.cuda.empty_cache()
print(f'Average Time: {np.mean(times):.2f} seconds')
print(f'Var Time: {np.var(times):.2f} seconds')
print(f'Average Memory: {np.mean(memories):.2f} MB')
print(f'Average Best Val Acc: {np.mean(best_val_accs):.4f}')
print(f'Std Best Test Acc: {np.std(best_test_accs):.4f}')
print(f'Average Test Acc: {np.mean(best_test_accs):.4f}')

