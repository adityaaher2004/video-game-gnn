import json
import pandas as pd

genre_occurrences = {}
total_genres = 0
total_occurrences = 0

fname = "data/games_metadata.json"
with open(fname, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data = json.loads(line)
            tags = data["tags"]
            name = data["app_id"]
            for genre in tags:
                if genre not in genre_occurrences:
                    total_genres += 1
                    genre_occurrences[genre] = 1
                else:
                    genre_occurrences[genre] += 1
                total_occurrences += 1
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

print(f"Total Genres = ", total_genres)
print(f"Total Occurrences = ", total_occurrences)

fname = "games_processed.csv"
df = pd.read_csv(fname)
print()
print("Data Frame Information:")
df_features = df.info()
print(df_features)
