import numpy as np
import pandas as pd

from scipy.spatial import distance
from statistics import mode

class KNeighbors:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.calc_distance = self.euclidean_distance
        self.training_dataframe = None
        
    
    @staticmethod
    def euclidean_distance(x,y):
        return distance.euclidean(x,y)
    
    @staticmethod
    def manhattan_distance(x,y):
        return distance.cityblock(x,y)
    
    def set_training_data(self, trainig_data):
        self.training_dataframe = trainig_data
        
    def set_distance_type(self,distance_type):
        if distance_type == "euclidean":
            self.calc_distance = self.euclidean_distance
        elif distance_type == "manhattan":
            self.calc_distance = self.manhattan_distance
        else:
            print("Wrong distance, using default (euclidean)")
            
    def classify(self,new_object):                
        
        nearest_neighbors= [[float('inf'),None] for _ in range(self.n_neighbors)]

        for _class, _object in self.training_dataframe.iterrows():

            neighbor_distance = self.calc_distance(new_object,_object.values)

            nearest_neighbors.sort(key=lambda dist: dist[0], reverse=True)
            if neighbor_distance < nearest_neighbors[0][0]:
                nearest_neighbors[0][0] = neighbor_distance
                nearest_neighbors[0][1] = _class
        
        # print(nearest_neighbors)
        nearest_neighbors = [_class[1] for _class in nearest_neighbors]
        

        return mode(nearest_neighbors)    
