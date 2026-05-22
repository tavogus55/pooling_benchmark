import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import torch
# from transformers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
import os
import random
import pandas as pd
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
import pandas as pd
from torch_geometric.datasets import GitHub
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time
import tracemalloc
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from sklearn.model_selection import KFold
import numpy as np
import random
from typing import Callable, Optional, Union
import time
# import psutil
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
max_nodes = 150
dataset_sparse = PygNodePropPredDataset(root='/data/XXX/Pooling', name="ogbn-arxiv")
evaluator = Evaluator("ogbn-arxiv")
eval_metric = dataset_sparse.eval_metric

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import torch
# from transformers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
import os
import random
import pandas as pd
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
import pandas as pd
from torch_geometric.datasets import GitHub
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time
import tracemalloc
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from sklearn.model_selection import KFold
import numpy as np
import random
from typing import Callable, Optional, Union
import time
# import psutil
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
max_nodes = 150
dataset_sparse = PygNodePropPredDataset(root='/data/XXX/Pooling', name="ogbn-arxiv")
evaluator = Evaluator("ogbn-arxiv")
eval_metric = dataset_sparse.eval_metric

from torch_geometric.datasets import GitHub
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time
import tracemalloc
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from sklearn.model_selection import KFold
import numpy as np
import random
from typing import Callable, Optional, Union
dataset = dataset_sparse
graph = dataset[0]
num_classes = dataset.num_classes
in_channels = dataset.num_features
hidden_channels = 64
out_channels = num_classes
depth = 2
pool_ratios = [0.5, 0.5]
class HierarchicalGCN_TopK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth, act=F.relu, sum_res=False):
        super(HierarchicalGCN_TopK, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = pool_ratios
        self.sum_res = sum_res
        self.act = act
        channels = self.hidden_channels
        self.pools = torch.nn.ModuleList()
        self.down_convs = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(self.in_channels, channels))
        for i in range(self.depth):
            self.pools.append(TopKPooling(channels, ratio=0.9))
            self.down_convs.append(GCNConv(channels, channels))
        in_channels = channels if sum_res else 2 * channels
        self.up_convs = torch.nn.ModuleList()
        for i in range(self.depth):
            self.up_convs.append(GCNConv(in_channels, channels))
        self.up_convs.append(GCNConv(channels, self.out_channels))
    def forward(self, x, edge_index, batch=None):
        x, edge_index = x.to(device), edge_index.to(device)
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if batch is not None:
            batch = batch.to(device)
        x = F.dropout(x, p=0.0, training=self.training)
        x = self.down_convs[0](x, edge_index)
        x = F.relu(x)
        xs = [x]
        edge_indices = [edge_index]
        for i in range(1, self.depth + 1):
            x, edge_index, _, batch, perm, _ = self.pools[i - 1](x, edge_index, batch=batch)
            x = self.down_convs[i](x, edge_index)
            x = F.relu(x)
            if i < self.depth:
                xs.append(x)
                edge_indices.append(edge_index)
        for i in range(self.depth):
            j = self.depth - 1 - i
            res = xs[j]
            edge_index = edge_indices[j]
            up = res
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)
            x = self.up_convs[i](x, edge_index)
            x = F.relu(x)
        x = self.up_convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)
def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()
@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()
    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    return train_acc, valid_acc, test_acc
seeds = [42, 123, 456]
results = []
val_accuracies_list = []
times = []
memories = []
gpu_memories = []
for seed in seeds:
    graph = graph.to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    val_accuracies = []
    test_accuracies = []
    start_time = time.time()
    tracemalloc.start()
    model = HierarchicalGCN_TopK(
        in_channels=graph.num_features,
        hidden_channels=256,
        out_channels=dataset.num_classes,
        depth=2,
        act=F.relu,
        sum_res=False
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    for epoch in range(1, 501):
        loss = train(model, graph, split_idx['train'], optimizer)
        train_acc, valid_acc, test_acc = test(model, graph, split_idx, evaluator)
        val_accuracies.append(valid_acc)
        test_accuracies.append(test_acc)
        if epoch % 10 == 0:
            print(f'Seed: {seed:03d}, Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {valid_acc:.4f}, Test Acc: {test_acc:.4f}')
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_time = end_time - start_time
    memory_usage = current / 10**6
    peak_memory_usage = peak / 10**6
    if torch.cuda.is_available():
        gpu_memory_usage = torch.cuda.max_memory_allocated(device) / 1024**2
        torch.cuda.reset_max_memory_allocated(device)
    else:
        gpu_memory_usage = 0
    val_accuracies_list.append(val_accuracies)
    times.append(elapsed_time)
    memories.append(memory_usage)
    gpu_memories.append(gpu_memory_usage)
    best_val_acc = max(val_accuracies)
    best_val_index = val_accuracies.index(best_val_acc)
    corresponding_test_acc = test_accuracies[best_val_index]
    results.append({
        'seed': seed,
        'best_val_acc': best_val_acc,
        'corresponding_test_acc': corresponding_test_acc,
        'elapsed_time': elapsed_time,
        'memory_usage': memory_usage,
        'gpu_memory_usage': gpu_memory_usage
    })
avg_val_acc = np.mean([result['best_val_acc'] for result in results])
avg_test_acc = np.mean([result['corresponding_test_acc'] for result in results])
avg_time = np.mean(times)
avg_memory = np.mean(memories)
avg_gpu_memory = np.mean(gpu_memories)
val_acc_variance = np.var([result['best_val_acc'] for result in results])
test_acc_variance = np.var([result['corresponding_test_acc'] for result in results])
time_variance = np.var(times)
memory_variance = np.var(memories)
gpu_memory_variance = np.var(gpu_memories)
for result in results:
    print(f"Seed: {result['seed']}, Best Val Acc: {result['best_val_acc']:.4f}, "
          f"Corresponding Test Acc: {result['corresponding_test_acc']:.4f}, "
          f"Time: {result['elapsed_time']:.2f}s, Memory: {result['memory_usage']:.2f}MB, "
          f"GPU Memory: {result['gpu_memory_usage']:.2f}MB")
print(f"\nAverage Best Validation Accuracy: {avg_val_acc:.4f}")
print(f"Average Corresponding Test Accuracy: {avg_test_acc:.4f}")
print(f"Average Time: {avg_time:.2f}s")
print(f"Average Memory Usage: {avg_memory:.2f}MB")
print(f"Average GPU Memory Usage: {avg_gpu_memory:.2f}MB")
print(f"\nValidation Accuracy Variance: {val_acc_variance:.4f}")
print(f"Test Accuracy Variance: {test_acc_variance:.4f}")
print(f"Time Variance: {time_variance:.4f}")
print(f"Memory Usage Variance: {memory_variance:.4f}MB")
print(f"GPU Memory Usage Variance: {gpu_memory_variance:.4f}MB")

