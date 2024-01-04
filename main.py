import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())
def Dim_Reduction(data):
    size = data.shape


