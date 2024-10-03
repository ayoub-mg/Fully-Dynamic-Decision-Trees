from typing import List, Tuple
from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""

    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
       split_feature : int
            along which feature the points have been split
        split_value: float
            Value of the split_feature based on which the splitting criteria was described. 
            For categorical features, the split is determined by the true class and false class values.
            For real features, points are split based on whether they are less than or greater than the split_value for the true class.
            For boolean features, the split_value is set to None (or 1.0)."
    """

    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """

        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.split = None
        self.threshold = None
        self.split_feature = None
        self.category_id = None
    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """

        n1,n2=0,0
        n=len(self.labels) 
        for label in self.labels :
            if label :
                n1+=1
            else :
                n2+=1
        return 1-(n1/n)**2-(n2/n)**2

    def gini_split(self, p: int, value: float = None, min_split_points: int = 1) -> float:
        """Compute the Gini split of the set of points based on the given value with a stopping rule

        Parameters
        ----------
        p : int
            Index of the feature in the features array based on which the split is made
        value : float
            Value of the feature that constitutes the splitting criteria
        min_split_points : int
            Lower bound of the number of points in the nodes (threshold). Each node in the
            tree (including the leaves) is associated with at least min_split_points points.

        Returns
        -------
        float
            The Gini split of the set of points
        """

        features1, features2 = [], []
        n=len(self.features)

        for i in range (n) :
            if self.types[p] == FeaturesTypes.BOOLEAN :
                if self.features[i][p] : features1+=[i]
                else : features2+=[i]
            elif self.types[p] == FeaturesTypes.CLASSES :
                if self.features[i][p] == value : features1+=[i]
                else : features2+=[i]
            elif self.types[p] == FeaturesTypes.REAL :
                if self.features[i][p] < value : features1+=[i]
                else : features2+=[i]

        n1 = sum(1 for k in features1 if self.labels[k])
        n2 = sum(1 for k in features2 if self.labels[k])
        n3 = sum(1 for k in features1 if not self.labels[k])
        n4 = sum(1 for k in features2 if not self.labels[k])

        if n1+n3==0 or n2+n4==0 or n1+n3<min_split_points or n2+n4<min_split_points : 
            return None

        gini1 = 1-(n1/(n1+n3))**2-(n3/(n1+n3))**2
        gini2 = 1-(n2/(n2+n4))**2-(n4/(n2+n4))**2

        return ((n1+n3)*gini1+(n2+n4)*gini2)/n
    
    def get_best_gain(self, min_split_points: int = 1) -> Tuple[int, float]:
        """Compute the best Gini Gain of the set of points with a stopping criteria 

        Parameters
        ----------
        min_split_points : int
            Lower bound of the number of points in the nodes (threshold). Each node in
            the tree (including the leaves) is associated with at least min_split_points points.

        Returns
        -------
        Tuple[int,float]
            Index of the feature along which splitting provides the best Gini gain,
            as well as the value of this gain.
        """

        max = None,0.0
        n_features=len(self.types)

        for p in range(n_features):
            values = np.unique(self.features[:,p])

            if self.types[p] == FeaturesTypes.BOOLEAN :
                split_values = [None]
            elif self.types[p] == FeaturesTypes.REAL :
                split_values = [(sorted(values)[i] + sorted(values)[i + 1]) / 2 for i in range(len(sorted(values)) - 1)]
            elif self.types[p] == FeaturesTypes.CLASSES : 
                split_values = values

            for v in split_values:
                gini_split = self.gini_split(p, v, min_split_points)
                if gini_split is not None:
                    gain = self.get_gini() - gini_split
                    if gain > max[1]:
                        max = p,gain
                        self.split_value = v
                        self.split_feature = p

        if max[1] == 0.0:
            return None, None

        return max    
                

    def get_best_threshold(self) -> float:
        """Compute the threshold of the selected feature that provides the best gain

        Returns
        -------
        float
            The best threshold along the feature that provides the best gain
        """

        if self.split_feature is None :
            raise ValueError("The method get_best_threshold() should be called after get_best_gain()!")
        
        return self.split_value     