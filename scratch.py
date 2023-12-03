import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# path = "data/games_processed.csv"
# data = pd.read_csv(path)
# for index, game in tqdm(data.iterrows(), total=data.shape[0]):
#     print(index)
#     print((game.array[0]))
#     break

fname = "data/games_metadata.json"
df_games_meta = pd.read_json(fname, lines=True, orient="records")
data = pd.Series(df_games_meta.loc[0, ['tags']])
genres = data.iloc[0]
for genre in genres:
    print(genre)
