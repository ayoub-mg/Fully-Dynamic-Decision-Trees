from typing import List
import numpy as np
from PointSet import PointSet, FeaturesTypes

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
        h : int
            The maximum height of the tree
        counter : int
            The counter of points added or removed from the tree
        feature_id : int
            The id of the feature used to split the tree
        feature_threshold : float
            The threshold used to split the tree. Is set to None if the feature is not continous
        feature_split : Tuple[List[float], List[float]]
            The split used to split the tree. Is set to None if the feature is continous
        left : Tree
            The left child of the tree
        right : Tree
            The right child of the tree
        min_split_points : int
            The minimum number of points required to split a node
        beta : float
            The regularization parameter. It is used to penalize the
            complexity of the tree. The higher the value of beta, the
            simpler the tree will.
    """

    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1,
                 min_split_points: int = 1,
                 beta : float = 0):
        """
        Parameters
        ----------
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            labels : List[bool]
                The labels of the training points.
            types : List[FeaturesTypes]
                The types of the features.
            h : int
                The maximum height of the tree.
            min_split_points : int
                The minimum number of points required to split a node.
            beta : float
                The regularization parameter.
        """

        self.h = h
        self.feature_id = None
        self.feature_threshold = None
        self.feature_split = None
        self.points = PointSet(features, labels, types)
        self.min_split_points = min_split_points
        self.counter = 0 
        self.beta = beta
        self.build_tree()

    def build_tree(self):
        """Recursively builds the decision tree by splitting points based on the best Gini gain."""

        # Stopping condition: Maximum depth reached
        if self.h == 0:
            return
        
        # Find the best feature and gain for splitting
        feature_id, best_gain = self.points.get_best_gain(min_split_points=self.min_split_points)

        # If no valid split is found, stop further splitting
        if feature_id is None:
            self.h = 0
            return

        # Determine if the feature is continuous or categorical
        is_continous = (self.points.types[feature_id] == FeaturesTypes.REAL)
        threshold = self.points.get_best_threshold()

        # Handle categorical or continuous splitting
        if not is_continous:
            if threshold is not None:
                split = ([threshold], list(set([f for f in self.points.features[:, feature_id] if f != threshold])))
            else:
                split = ([0], [1])
            self.feature_split = split
        else:
            self.feature_threshold = threshold

        self.feature_id = feature_id

        # Prepare lists to store the left and right splits
        left_features, left_labels = [], []
        right_features, right_labels = []

        # Split the points into left and right subsets based on the threshold or split
        for i, point in enumerate(self.points.features):
            if (is_continous and point[self.feature_id] < threshold) or (not is_continous and point[self.feature_id] in split[0]):
                left_features.append(point)
                left_labels.append(self.points.labels[i])
            else:
                right_features.append(point)
                right_labels.append(self.points.labels[i])

        # If either left or right has too few points, stop further splitting
        if len(left_features) < self.min_split_points or len(right_features) < self.min_split_points:
            self.h = 0
            return

        # Recursively build the left and right subtrees
        self.left = Tree(left_features, left_labels, self.points.types, self.h - 1, self.min_split_points, self.beta)
        self.left.build_tree()

        self.right = Tree(right_features, right_labels, self.points.types, self.h - 1, self.min_split_points, self.beta)
        self.right.build_tree()

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
        features : List[float]
            The features of the unlabeled point.

        Returns
        -------
        bool
            The label of the unlabeled point, guessed by the Tree.
        """
        
        # If the node is a leaf, return the majority label in this node
        if self.h == 0:
            # Determine the majority class (True or False) in the leaf
            return np.count_nonzero(self.points.labels) >= len(self.points.labels) / 2

        # Handle continuous and categorical splits
        if self.feature_split is None:  # Continuous feature split
            if features[self.feature_id] < self.feature_threshold:
                return self.left.decide(features)
            else:
                return self.right.decide(features)
        else:  # Categorical feature split
            if features[self.feature_id] in self.feature_split[0]:
                return self.left.decide(features)
            else:
                return self.right.decide(features)

    def add_training_point(self, features: List[float], label: bool):
        """Add a new training point to the tree, potentially rebuilding it if necessary."""
        
        # Increment the counter of points added
        self.counter += 1

        # Add the new point to the PointSet only if the tree is not a leaf
        if self.h > 0:
            new_features = np.vstack([self.points.features, features])
            new_labels = np.append(self.points.labels, label)
            self.points = PointSet(new_features, new_labels, self.points.types)

        # Determine if the tree needs to be rebuilt based on the regularization parameter beta
        if self.counter >= self.beta * len(self.points.features):
            self.counter = 0
            
            # Rebuild the tree
            self.build_tree()
            return

        # If the current node is a leaf, do nothing
        if self.h == 0:
            return

        # Recursively add the training point to the correct subtree
        if self.feature_split is None:  # Continuous feature split
            if features[self.feature_id] < self.feature_threshold:
                self.left.add_training_point(features, label)
            else:
                self.right.add_training_point(features, label)
        else:  # Categorical feature split
            if features[self.feature_id] in self.feature_split[0]:
                self.left.add_training_point(features, label)
            else:
                self.right.add_training_point(features, label)    

    def del_training_point(self, features: List[float], label: bool):
        """Delete a training point from the tree and rebuild the tree if necessary."""
        
        # Increment the counter of points removed
        self.counter += 1

        # Find the index of the point to remove
        if self.h > 0:
            # Convert features to a numpy array for comparison
            features_array = np.array(features)
            
            # Use a boolean mask to find the point matching the features
            mask = np.all(self.points.features == features_array, axis=1)
            indices_to_remove = np.where(mask)[0]
            
            # If no matching point is found, exit the function
            if len(indices_to_remove) == 0:
                return

            # Remove the point and its label using numpy indexing
            new_features = np.delete(self.points.features, indices_to_remove[0], axis=0)
            new_labels = np.delete(self.points.labels, indices_to_remove[0], axis=0)

            # Update the PointSet with the remaining points
            self.points = PointSet(new_features, new_labels, self.points.types)

        # Rebuild the tree if necessary based on the beta regularization parameter
        if self.counter >= self.beta * len(self.points.features):
            self.counter = 0
            self.build_tree()
            return

        # If the current node is a leaf, stop the process
        if self.h == 0:
            return

        # Recursively delete the point in the correct subtree
        if self.feature_split is None:  # Continuous feature split
            if features[self.feature_id] < self.feature_threshold:
                self.left.del_training_point(features, label)
            else:
                self.right.del_training_point(features, label)
        else:  # Categorical feature split
            if features[self.feature_id] in self.feature_split[0]:
                self.left.del_training_point(features, label)
            else:
                self.right.del_training_point(features, label)