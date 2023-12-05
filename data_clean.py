import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

fname = "data/games_test.csv"
df = pd.read_csv(fname)
total_rows = 400

# Normalizing user reviews
review_column = 'user_reviews'
data = df[review_column]
data = np.array(data)
scaler = MinMaxScaler()
scaler.fit(data.reshape(-1,1))
scaled_data = scaler.fit_transform(data.reshape(-1,1))
df[review_column] = scaled_data

# Converting user rating from String to int
rating_column = 'rating'
rating_convert_dict = {'Overwhelmingly Positive' : 9.8,
                        'Very Positive' : 9,
                        'Positive' : 8.5,
                        'Mostly Positive' : 7.5,
                        'Mixed' : 5.25,
                        'Mostly Negative' : 3,
                        'Negative' : 2,
                        'Very Negative' : 1.5,
                        'Overwhelmingly Negative' : 0.5}
rating_data = df[rating_column]
new_rating = []
for rating in rating_data:
    if rating not in rating_convert_dict.keys():
        print(f"Rating not present : ", rating)
        rating = 8
        continue
    new_rating.append(rating_convert_dict[rating])
df[rating_column] = new_rating
df = df[:800]
df.to_csv("data/raw/games_processed.csv")

