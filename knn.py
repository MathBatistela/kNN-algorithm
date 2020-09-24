import numpy as np
import pandas as pd

from scipy.spatial import distance
from statistics import mode

class KNeighbors:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        
    training_dataframe = None
    calc_distance = distance.euclidean
    
    def set_training_data(self, trainig_data):
        self.training_dataframe = trainig_data
        
    def set_distance_type(self,distance_type):
        self.distance_type = distance_type
    
    def classify(self,new_object):                
        
        nearest_neighbors= [[float('inf'),None] for _ in range(3)]

        for _class, _object in self.training_dataframe.iterrows():

            neighbor_distance = distance.euclidean(new_object,_object.values)

            nearest_neighbors.sort(key=lambda dist: dist[0], reverse=True)
            for value in nearest_neighbors:
                if neighbor_distance < value[0]:
                    value[0] = neighbor_distance
                    value[1] = _class
                    break
        
        nearest_neighbors = [_class[1] for _class in nearest_neighbors]
        
        return mode(nearest_neighbors)    
