import torch
from torch_geometric.data import Dataset, Data
from torch_geometric import utils
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import networkx as nx

class GamesDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root -> root directory of data
        raw_dir -> downloaded dataset
        processed_dir -> processed data
        """
        self.test = test
        self.filename = filename
        self.node_mapping = {}
        super(GamesDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """
        if file exists, download is not triggered
        """
        return self.filename
    
    @property
    def processed_file_names(self):
        """
        process skipped if file not found
        """
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        if self.test:
            return [f'data_test_{i}' for i in list(self.data.index)]
        else:
            return [f'data_{i}' for i in list(self.data.index)]

    
    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])

        for index, game in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            self.node_mapping[game.array[1]] = index            
        for index, game in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            self.node_mapping[game[0]] = index

            # Get node Features and target labels
            node_features = self._get_node_features(game)
            target_labels = self._get_target_labels(game, index)
            edge_list, edge_attr = self._get_edge_features(game, index)

            data = Data(x=node_features,
                        edge_index=edge_list.t().contiguous(),
                        edge_attr=edge_attr,
                        y=target_labels)

            if self.test:
                torch.save(data,
                        os.path.join(self.processed_dir, f'data_test_{index}.pt'))
            else:
                torch.save(data,
                        os.path.join(self.processed_dir, f'data_{index}.pt'))
            
    def _get_node_features(self, node):
        """
        Returns array of node features
        Features:
        - Rating
        - Year Released
        - Postitive Ratio
        - Normalized number of reviews
        """
        node_features = []

        for ind in range(self.data.shape[0]):
            node = self.data.loc[ind, :]    
            game = node
            year_released = int((game.array[3])[-4:])
            rating = game.array[7]  
            positive_ratio = game.array[8]
            reviews = game.array[9]
            features = [year_released, rating, positive_ratio, reviews]
            node_features.append(features)

        return torch.tensor(node_features, dtype=torch.float)
    
    def _get_target_labels(self, node, index):
        target_labels = []
        game = node.array
        node_id = game[0]
        node_ratio = game[6]
        node_rating = game[5]
        node_year = int((game[3])[-4:])
        fname = "data/raw/games_metadata_test.json" if self.test else "data/raw/games_metadata.json"
        df_games_meta = pd.read_json(fname, lines=True, orient="records")
        node_genres = (pd.Series(df_games_meta.loc[index, ['tags']])).iloc[0]

        for ind in range(self.data.shape[0]):
            if ind == index:
                continue
            else:
                target_ratio = self.data.loc[ind, "positive_ratio"]
                target_rating = self.data.loc[ind, "rating"]

                # ... (you can add more conditions based on your criteria)

                # Define your criteria for positive and negative links
                target_genres_data = pd.Series(df_games_meta.loc[ind, ['tags']])
                target_genres = target_genres_data.iloc[0]
                if len(target_genres) >= 20:
                    target_genres = target_genres[:20]
                set_target = set(target_genres)
                set_node = set(node_genres)
                flag = len(set_node.intersection(set_target))
                if flag > 0:
                    target_labels.append(1)  # Positive link
                else:
                    target_labels.append(0)  # Negative link

        return torch.tensor(target_labels, dtype=torch.float)
    
    def _get_edge_features(self, node, index):
        """
        Returns Adjacency List(CCO format) as well as Edge Features
        """
        edge_list = []
        edge_attr = []
        game = node.array
        node_id = game[0]
        node_ratio = game[6]
        node_rating = game[5]
        node_year = int((game[3])[-4:])
        fname = "data/raw/games_metadata_test.json" if self.test else "data/raw/games_metadata.json"
        df_games_meta = pd.read_json(fname, lines=True, orient="records")
        node_genres = (pd.Series(df_games_meta.loc[index, ['tags']])).iloc[0]
        num = self.data.shape[0]
        counter = 0
        for ind in range(num):
            if ind == index:
                continue
            else:
                target_ratio = self.data.loc[ind, "positive_ratio"]
                target_rating = self.data.loc[ind, "rating"]
                target_id = self.data.loc[ind, "app_id"]
            
                target_genres_data = pd.Series(df_games_meta.loc[ind, ['tags']])
                target_genres = target_genres_data.iloc[0]
                if len(target_genres) >= 10:
                    target_genres = target_genres[:10]
                set_target = set(target_genres)
                set_node = set(node_genres)
                flag = len(set_node.intersection(set_target))
                if flag > 0:
                    edge_list.append([self.node_mapping[node_id], self.node_mapping[target_id]])
                edge_attr.append([flag])

                
        
        return torch.tensor(edge_list, dtype=int), torch.tensor(edge_attr, dtype=int)
    
    def len(self):
      return self.data.shape[0]

    def get(self, idx):
      return torch.load(os.path.join(self.processed_dir,f"data_{idx}.pt"))
    