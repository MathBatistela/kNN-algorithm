import sys
import numpy as np
import pandas as pd
import knn as knn

from scipy.spatial import distance
from scipy import stats

def min_max(a):
    return (a - a.min())/(a.max() - a.min())

def normalize(data_matrix,normalization):
    df = pd.DataFrame(data_matrix,index=data_matrix[:,-1].astype(int))
    df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
    return df.apply(normalization , axis=1).copy()

if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Please, provide two files as arguments")
        sys.exit()
        
    try:
        training_matrix = np.loadtxt(sys.argv[1], delimiter=' ')
        test_matrix = np.loadtxt(sys.argv[2], delimiter=' ')
    except OSError:
        print("File does not exist")
        
   
    training_norm_df = normalize(training_matrix,min_max)
    test_norm_df = normalize(test_matrix,min_max)

    
    _object = test_norm_df.iloc[101].values
    
    _knn = knn.KNeighbors(n_neighbors=3)
    
    _knn.set_training_data(training_norm_df)
    
    print(_knn.classify(_object))
    
    