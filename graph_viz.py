import os.path as osp
import torch
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.nn import GraphConv
from torch_geometric.utils import negative_sampling
from generate_dataset import GamesDataset
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
device = torch.device('cpu')

path = osp.join(osp.dirname(osp.realpath(__name__)), 'data')
dataset = GamesDataset(path, "games_processed.csv")
# Instead of using the transform, split the dataset manually

total_edges = 0
max_edges = 0
data = dataset[1]  # Get the first graph object.

for node in dataset:
    total_edges += node.num_edges
    if node.num_edges > max_edges:
        max_edges = node.num_edges

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of Nodes: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f"Total Edges = {total_edges}")
print('======================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Average node degree: {total_edges / data.num_nodes:.2f}')
print(f'Maximum node degree: {max_edges}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
print('======================')

# create_graph()


def create_graph():
    G = nx.Graph()

    for idx in range(len(dataset)):
        # Choose a specific graph from the dataset
        sample_graph = dataset[idx]

        # Extract relevant information from the PyTorch Geometric data structure
        edge_index = sample_graph.edge_index
        node_features = sample_graph.x

        # Convert the PyTorch Geometric data to a NetworkX graph
        G.add_edges_from(edge_index.t().numpy())

        # Calculate node degrees
        node_degrees = dict(G.degree())
        max_degree = max(node_degrees.values()) if node_degrees.values() else 1

        # Normalize degrees for heat map
        normalized_degrees = {node: degree / max_degree for node, degree in node_degrees.items()}

        # Calculate spectral clustering
        spectral = SpectralClustering(n_clusters=20, affinity='nearest_neighbors', random_state=42)
        clusters = spectral.fit_predict(node_features.numpy())

    # Visualize the NetworkX graph with heat maps and clustering

    # Heat map for node degrees
    nx.draw_networkx_nodes(G, pos=nx.spring_layout(G), cmap=plt.cm.Reds, node_color=list(normalized_degrees.values()), node_size=30)
    nx.draw_networkx_edges(G, pos=nx.spring_layout(G), edge_color="gray")

    # Clustering using color-coded nodes
    nx.draw_networkx_nodes(G, pos=nx.spring_layout(G), cmap=plt.cm.tab10, node_color=clusters[:G.number_of_nodes()], node_size=30)

    # Add game titles as labels

    # node_labels = {node: dataset._get_id_name(node) for node in G.nodes()}
    # nx.draw_networkx_labels(G, pos=nx.spring_layout(G), labels=node_labels, font_size=8)


    sm = plt.cm.ScalarMappable(cmap=plt.cm.tab10)
    sm.set_array([])  # Set an empty array

    # Add colorbar with explicitly specified axes
    cbar = plt.colorbar(sm, ax=plt.gca(), label="Cluster")
    cbar.set_ticks([])  # You can customize tick positions if needed

    plt.title(f"Graph Visualization - Index {idx}")
    plt.show()
