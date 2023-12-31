import os.path as osp
import torch
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.nn import GraphConv
from torch_geometric.utils import negative_sampling
from generate_dataset import GamesDataset
import pandas as pd
import plotly.graph_objects as go
device = torch.device('cpu')

# Remove the NormalizeFeatures transform
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
    add_negative_train_samples=True),
])

path = osp.join(osp.dirname(osp.realpath(__name__)), 'data')
dataset = GamesDataset(path, "games_processed.csv", transform=transform)
# Instead of using the transform, split the dataset manually
train_data, val_data, test_data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

model = Net(dataset.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='dense')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )

    edge_label = torch.cat([train_data.edge_label, train_data.edge_label.new_zeros(neg_edge_index.size(1))], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss
ep = []
val_accuracy = []
test_accuracy = []

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

best_val_auc = final_test_auc = 0
for epoch in range(1, 101):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    ep.append(epoch)
    val_accuracy.append(val_auc*100)
    test_accuracy.append(test_auc*100)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
            f'Test: {test_auc:.4f}')

print(f'Final Test: {final_test_auc:.4f}')

z = model.encode(test_data.x, test_data.edge_index)
final_edge_index = model.decode_all(z)

df = pd.DataFrame(dict(
    x = ep,
    y1 = test_accuracy,
    y2 = val_accuracy
))

fig = go.Figure()
fig.add_trace(go.Scatter(x=ep, y=val_accuracy,
                    mode='lines+markers',
                    name='Validation Accuray'))

fig.add_trace(go.Scatter(x=ep, y=test_accuracy,
                    mode='lines+markers',
                    name='Test Accuray',))

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.88
))

fig.update_xaxes(range=[1, 70], )
fig.update_yaxes(range=[-5,100])

fig.show()

