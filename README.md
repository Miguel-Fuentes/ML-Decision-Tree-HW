# ML-Decision-Tree-HW

## Run this command to train a decision tree with ID3 algorithm and the resulting output will consists of accuracies of entropy and impurity heuristics along with the corresponding trees if print option is selected

`python ID3_Tree.py 5 10 data_sets1/training_set.csv data_sets1/validation_set.csv data_sets1/test_set.csv yes`


## Pruning Results

### Data set 1
| Entropy ACC | Impurity ACC | K  | L  |
|:-----------:|--------------|----|----|
|    0.723    |     0.725    | 0  | 0  |
|    0.723    |     0.735    | 3  | 10 |
|    0.703    |     0.73     | 7  | 10 |
|    0.735    |    0.7183    | 11 | 10 |
|    0.723    |     0.726    | 15 | 10 |
|    0.725    |     0.716    | 19 | 10 |
|    0.733    |     0.74     | 3  | 20 |
|     0.76    |     0.726    | 7  | 20 |
|    0.726    |     0.74     | 11 | 20 |
|    0.725    |     0.741    | 15 | 20 |
|    0.723    |     0.715    | 19 | 20 |
