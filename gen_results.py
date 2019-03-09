import pandas as pd
import decision_tree

data_folders = ['data_sets1/','data_sets2/']

for folder in data_folders:
    print(f'Reading data from {folder}')
    training_set = pd.read_csv(folder + 'training_set.csv')
    validation_set = pd.read_csv(folder + 'validation_set.csv')
    test_set = pd.read_csv(folder + 'test_set.csv')
    
    # Initialize the two trees
    attributes = training_set.columns.to_list()
    attributes.remove('Class')

    tree1 = decision_tree.Node(attributes, [], training_set, decision_tree.entropy_gain)
    tree2 = decision_tree.Node(attributes, [], training_set, decision_tree.impurity_gain)
    print('Trees Initialized, trees will train now this may take up to 5 minutes')

    tree1.train()
    tree2.train()
    print('Trees Trained')

    results = []
    results.append({'L':0, 'K': 0,
                    'Entropy Acc' : decision_tree.accuracy(tree1, test_set),
                    'Impurity Acc' : decision_tree.accuracy(tree2, test_set)})
    
    for L in [10,20]:
        for K in [3, 7, 11, 15, 19]:
            pruned1 = tree1.post_pruning(L, K, validation_set)
            pruned2 = tree2.post_pruning(L, K, validation_set)
            
            results.append({'L':0, 'K': 0,
                            'Entropy Acc' : decision_tree.accuracy(pruned1, test_set),
                            'Impurity Acc' : decision_tree.accuracy(pruned2, test_set)})
    results_df = pd.DataFrame(results)
    results_df.to_csv(folder + 'pruning_test_results.csv')
    print(f'Results saved to {folder}pruning_test_results.csv')
    