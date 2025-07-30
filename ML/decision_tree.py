import numpy as np
from collections import Counter

'''
(information gain) IG = entropy(parent)-[weighted average]*entropy(children)
    here, entropy is randomness in the specific node. 
    range: 0-1
    nodes where values are 50-50, entropy = 1 (max possible)

(entropy) E = - sigma(p(X)*log2(p(X)))
    here, p(X)= #x/n (number of times a class has occured in the node/number of total nodes)

Stopping critera: maximum depth
'''

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTrees:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        # min samples split sets the minimum samples there should be in a node. if less than min_samples_split, do not split
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None


    def fit(self, X, y):
        # if number of features is not defined
        self.n_features=X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root=self._grow_tree(X,y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats=X.shape
        n_labels=len(np.unique(y))

        # check stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value=self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs=np.random.choice(n_feats, self.n_features, replace=False)

        # find best split
        best_feature, best_thresh=self._best_split(X, y, feat_idxs)


    # HUH ??
    def _best_split(self, X, y, feat_idxs):
        best_gain=-1
        split_idx, split_threshold=None, None

        for feat_idx, in feat_idxs:
            X_column=X[:, feat_idx]
            thresholds=np.unique(X_column)

            for thr in thresholds:
                # calculate information gain
                gain=
                
                if gain>best_gain:
                    best_gain=gain
                    split_idx=feat_idx
                    split_threshold=thr

        return split_idx, split_threshold

    def _most_common_label(self, y):
        counter=Counter(y)
        # gives most common item's label ONLY
        value=counter.most_common(1)[0][0]
        return value

    def predict():
