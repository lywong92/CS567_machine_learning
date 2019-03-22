from __future__ import division, print_function

from typing import List

import numpy as np
import scipy

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        self.features = features
        self.labels = labels
        
    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        result = []
        # get labels using get_k_neighbors
        for x in range(len(features)):
            classes = {}
            knn = self.get_k_neighbors(features[x])
            for i in range(len(knn)):
                if knn[i] not in classes.keys():
                    classes[knn[i]] = 1
                else:
                    classes[knn[i]] += 1
            result.append(max(classes, key=classes.get))
        
        return result
    
    #TODO: find KNN ofeturn result one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        # account of corner test cases
        points = len(self.features)
        if points < 30:
            self.k = points if points%2 == 1 else points-1
        elif self.k < 1:
            self.k = 1

        knn = []
        neighbors = []
        # calcuate distance from input point to every other data point
        # sort distances in increasing order
        for n in range(len(self.features)):
            distance = self.distance_function(point, self.features[n])
            neighbors.append(distance)
        neighbors_sorted = np.argsort(neighbors)
        for x in range(self.k):
            knn.append(self.labels[neighbors_sorted[x]])
            
        return knn
             

if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
