# Leveraging Graph Neural Networks To Build A Video Game Recommender System


## Libraries Required

1. PyTorch : Tensor Data Loading
2. PyTorch Geometric : GNN Layers, Creating a Custom Dataset
3. Networkx and Matplotlib *(optional)* : Graph Visualization


## Dataset


* 1000 Nodes
* 750,000 Edges
* 750 Average Node Degree


### games.csv

##### Sample Data

* app id: 13500
* title: Prince of Persia: Warrior Within
* date released: 21-11-2008
* win: TRUE
* mac: FALSE
* linux: FALSE
* rating: Very Positive
* positive ratio: 84
* user reviews: 2199
* price final: 9.99
* price original: 9.99
* discount: 0
* steam deck: TRUE

------

### games_metadata.json

##### Sample Data

* "app id" : 13500, 
* "description" : "Enter the dark underworld, · · · preordained death."
* "tags" : ["Action","Adventure","Parkour", · · · "Puzzle"]

> ~ 500 unique genres (tags) with more than 580,000 cumulative occurences

------

## Model Evaluation

| Model | Validation Accuracy | Test Accuracy |
| ------------- | ------------- | ----- |
| Basic  | 80 | 60 |
| Edge Weights | 65 | 80 |
| Edge Weights and Negative Sampling | 84 | 94 |





