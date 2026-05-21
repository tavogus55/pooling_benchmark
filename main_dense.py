import time

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
import random
import numpy as np
from torch_geometric.data import DenseDataLoader
from dense_pooling_models import Net_AsymCheegerCut

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

max_nodes = 800
data_path = "data"
dataset_dense = TUDataset(
    data_path,
    name="PROTEINS",
    transform=T.Compose([T.ToDense(max_nodes)]),
    use_node_attr=True,
    pre_filter=lambda data: data.num_nodes <= max_nodes,
)

dataset = dataset_dense
dataset = dataset.shuffle()

model = Net_AsymCheegerCut(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.adj, data.mask)
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
        output = model(data.x, data.adj, data.mask)
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
    train_loader = DenseDataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DenseDataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DenseDataLoader(test_dataset, batch_size=128, shuffle=False)
    model = Net_AsymCheegerCut(dataset.num_features, dataset.num_classes).to(device)
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