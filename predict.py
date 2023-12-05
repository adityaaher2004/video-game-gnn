import torch
from torch_geometric.data import Data
from model import Net
from torch_geometric.data import Data

# Assuming you have 4 features for each node, 2 nodes, and 2 edges
x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float)
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
edge_attr = torch.tensor([[0.1], [0.2]], dtype=torch.float)
y = torch.tensor([1, 0], dtype=torch.float)
edge_label = torch.tensor([0.5, 0.8], dtype=torch.float)
edge_label_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

sample_data_point = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                         edge_label=edge_label, edge_label_index=edge_label_index)


def predict_links(model, data_point):
    model.eval()

    with torch.no_grad():
        # Assuming data_point is an instance of your GamesDataset class
        x = data_point.x
        edge_index = data_point.edge_index

        z = model.encode(x, edge_index)
        predicted_edges = model.decode_all(z)

    return predicted_edges

if __name__ == "__main__":
    # Load your trained model
    model_path = "model.py"
    params = {
    'in_channels': 16,
    'hidden_channels': 64,
    'out_channels': 32
    }

    loaded_model = Net(4, 128, 16).to('cpu')  # Replace 'your_parameters' with the actual model parameters
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    # Assuming you have a data point available (e.g., test_data_point)
    # Replace this with an actual instance of your GamesDataset
    test_data_point = ...

    # Get predicted edges
    predicted_edges = predict_links(loaded_model, test_data_point)

    # Output the first 10 predicted links
    print("Predicted Links:")
    for i in range(min(10, predicted_edges.size(1))):
        src, tgt = predicted_edges[:, i].tolist()
        print(f"Edge {i + 1}: Node {src} -> Node {tgt}")
