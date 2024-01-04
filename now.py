
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

def random_projection(csv_file, target_dim=2):
    # Load the CSV file into a Pandas DataFrame
    data = pd.read_csv(csv_file)
    
    # Extract the features
    features = data.iloc[:, :-1]  # Assuming the last column is the target variable
    
    # Randomly generate projection matrix
    num_features = features.shape[1]
    projection_matrix = np.random.randn(num_features, target_dim)
    
    # Project data onto the random subspace
    reduced_data = np.dot(features, projection_matrix)
    
    # Create a DataFrame with reduced dimensions
    reduced_df = pd.DataFrame(reduced_data, columns=[f'dim_{i+1}' for i in range(target_dim)])
    
    # Combine with the target variable if needed
    if len(data.columns) > len(features.columns):
        target_variable = data.iloc[:, -1]
        reduced_df['target'] = target_variable
    
    
    print(reduced_df.shape)
random_projection(df, 2)
