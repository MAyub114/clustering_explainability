import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from ExKMC.Tree import Tree


# Load dataset
iris = pd.read_csv("iris.csv")

print("Iris dataset:")
print(iris.head())
print("")

# Set up features X for K-means model
# Convert values to z-scores
# Drop "variety" column as it is the target
X = iris.drop('variety', axis=1)
for cols in X.columns:
    X[cols] = X[cols].astype(float)
    k1 = X[cols].mean()
    k2 = np.std(X[cols])
    X[cols] = (X[cols] - k1)/k2

# Show features
print("Features (X):")
print(X)

# K-means model 
k = 3
kmeans = KMeans(k, random_state=43)
kmeans.fit(X)
p = kmeans.predict(X)
class_names = np.array(['Setosa', 'Versicolor', 'Virginica'])

# IMM algorithm
# Creates decision tree with exactly k leaves
tree = Tree(k=k)
tree.fit(X, kmeans)

# Show decision tree
tree.plot(filename="iris_imm", feature_names=X.columns)

# ExKMC algorithm
# Creates decision tree with more than k leaves
tree = Tree(k=k, max_leaves=6)
tree.fit(X, kmeans)

# Show decision tree
tree.plot(filename="iris_exkmc", feature_names=X.columns)