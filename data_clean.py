import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

fname = "data/games.csv"
df = pd.read_csv(fname)
column = 'user_reviews'
data = df[column]
data = np.array(data)
scaler = MinMaxScaler()
scaler.fit(data.reshape(-1,1))
scaled_data = scaler.fit_transform(data.reshape(-1,1))
df[column] = scaled_data
df.to_csv("data/games_processed.csv")
