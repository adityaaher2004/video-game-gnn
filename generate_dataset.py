import torch as t
from torch_geometric.data import Dataset, Data
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

class GamesDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        root -> root directory of data
        raw_dir -> downloaded dataset
        processed_dir -> processed data
        """
        super(GamesDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """
        if file exists, download is not triggered
        """
        return "games_processed.csv"
    
    @property
    def processed_file_names(self):
        """
        process skipped if file not found
        """
        return "not_implemented.pt"
    
    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for index, game in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Get node Features
            node_features = self._get_node_features(game)

            ### 
            # Get edge features
            # Edges are formed if common genre exists between games
            # This will possibly make the graph dense
            # Total genres = 441
            # Total occurrences of all ~ 580,000
            ###
            edge_list = self._get_edge_features(game, index)

            data  = Data(x=node_features, 
                         edge_index=edge_list,
                         )

            t.save(data, 
                       os.path.join(self.processed_dir,
                                    f'data_{index}.pt'))
            
    def _get_node_features(self, node):
        """
        Returns array of node features
        Features:
        - Rating
        - Year Released
        - Postitive Ratio
        - Normalized number of reviews
        """

        game = node
        year_released = int((game.array[3])[-4:])
        rating = game.array[7]
        positive_ratio = game.array[8]
        reviews = game.array[9]

        node_features = np.asarray(list(year_released, rating, positive_ratio, reviews))
        return t.tensor(node_features, dtype=t.float)
    
    def _get_edge_features(self, node, index):
        """
        Returns Adjacency List(CCO format) as well as Edge Features
        """
        edge_list = []
        game = node.array
        id = game.array[0]
        df_games_meta = pd.read_json(self.raw_paths[1], lines=True, orient="records")
        node_genres = (pd.Series(df_games_meta.loc[index, ['tags']])).iloc[0]
        num = self.data.shape[1]
        for ind in range(num):
            if ind == index:
                continue
            else:
                target_genres_data = pd.Series(df_games_meta.loc[0, ['tags']])
                target_genres = target_genres_data.iloc[0]
                for genre in target_genres:
                    if genre in node_genres:
                        edge_list.append(index, ind)
                        break
        return t.tensor(edge_list, dtype=t.long)

print(t.__version__)
dataset = GamesDataset(root="data/")
print(dataset[0].edge_index.t())
print(dataset[0].x)
