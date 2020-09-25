import sys
import numpy as np
import pandas as pd
import knn as knn
import random

def min_max(attribute):
    return (attribute - attribute.min())/(attribute.max() - attribute.min())

def z_score(attribute):
    return (attribute - attribute.mean())/attribute.std()

def normalize(data_matrix,normalization):
    df = pd.DataFrame(data_matrix,index=data_matrix[:,-1].astype(int))
    df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
    return df.apply(normalization , axis=1).copy()

def result_file(instances):
    output = open('result.txt','w')
    for instance in instances:
        output.write("-----------------------------------------------------------------------\n")
        output.write(f"| K = {instance['k']} | Porcentagem: {instance['percent']*100} % | Normalização: {instance['normalization']} | Distância: {instance['distance']} |\n\n")
        cf = np.asarray(instance['confusion_matrix'])
        accuracy = (cf.trace()/cf.sum())*100
        output.write(f"| Taxa de acerto: {accuracy} %, erro: {100 - accuracy} % |\n\n")
        for _class in instance['confusion_matrix']:
            output.write(f"{_class}\n")
        output.write("\n")
        output.write("-----------------------------------------------------------------------\n")
        
def reduce_dataframe(dataframe, percent):
    if(percent == 1):
        return dataframe
    
    dataframe_indexes = dataframe.index.values
    dataframe_indexes = list(dict.fromkeys(dataframe_indexes))

    class_data = []
    for _index in dataframe_indexes:
        class_data.append(
            dataframe.groupby(
            dataframe.index).get_group(_index).sample(
            frac=percent,random_state=random.randrange(0,100)))

    return pd.concat(class_data)

if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Please, provide two files as arguments")
        sys.exit()
        
    try:
        training_matrix = np.loadtxt(sys.argv[1], delimiter=' ')
        test_matrix = np.loadtxt(sys.argv[2], delimiter=' ')
    except OSError:
        print("File does not exist")
        
    instances = []
        
    for percent in (0.25,0.5,1):  
          
        for normalization in (min_max,z_score):
            
            training_norm_df = reduce_dataframe(normalize(training_matrix,normalization),percent)
            test_norm_df = reduce_dataframe(normalize(test_matrix,normalization),percent)
            
            for distance in ("euclidean","manhattan"):
                
                for k in (1,3,5,7,9,11,13,15,17,19):
                    
                    _knn = knn.KNeighbors(n_neighbors=k)
                    _knn.set_training_data(training_norm_df)
                    _knn.set_distance_type(distance)   
                    
                    confusion_matrix = [ [ 0 for _ in range(10) ] for _ in range(10) ]
                    
                    for _class, _object in test_norm_df.iterrows():
                        classified_class = _knn.classify(_object.values)
                        confusion_matrix[_class][classified_class] += 1 
                        
                    instance = {
                        "k": k,
                        "percent": percent,
                        "normalization": normalization.__name__,
                        "distance": distance,
                        "confusion_matrix": confusion_matrix
                    }                    
                    
                    instances.append(instance)    
              
    
    result_file(instances)
    print("Done !")
