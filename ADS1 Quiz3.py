# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 21:24:59 2023

@author: kjide
"""

# Import necessary modules
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import err_ranges as err
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

# Define functions
def readfile(doc,columns,indicator):
    filedata = pd.read_excel(doc,skiprows=3)
    filedata = filedata.loc[filedata['Indicator Name'] == indicator]
    filedata = filedata[columns]
    filedata.set_index('Country Name', inplace = True)
    return filedata,filedata.transpose()

# Initialize attributes
doc = 'climate change.xls'
indicator = 'Agricultural land (% of land area)'
columns = ['Country Name', '1980','2019']

# Read and Print Data from file
Climate_Change = readfile(doc,columns,indicator)
print(Climate_Change)

# Extract the required data for the clustering
Climate_Change[0]

# Clean up data by removing all null values
ClimateChange = Climate_Change[0].dropna(axis=0)
print(ClimateChange)

# Normalize the data
scaler = preprocessing.MinMaxScaler()
Climate = scaler.fit_transform(ClimateChange)
print(Climate)

# Scatter plot of clusters
plt.scatter(Climate[:,0], Climate[:,1])
plt.xlabel('1980')
plt.ylabel('2019')
plt.title('Scatter Plot of Agricultural land (% of land area)')
plt.legend()
plt.show()

# Find the optimum number of clusters for k-means classification
sse = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', 
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(Climate)
    sse.append(kmeans.inertia_)

# Use the elbow method to determine the optimal number of clusters for k-means clustering.
plt.plot(range(1, 11), sse)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE') #Sum of Squared Error
plt.show()

# Determine number of clusters for Clustering
kmeans = KMeans(n_clusters = 3, init = 'k-means++', 
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(Climate)
print(y_kmeans)

# Creates a new dataframe with the labels for each attribute
ClimateChange['Cluster'] = kmeans.labels_ #add new column and in it number cluster
print(ClimateChange)

# Determine the centroids
centroids = kmeans.cluster_centers_
print(centroids)

# Use Scatter Plot to show clusters and centroids
plt.scatter(Climate[y_kmeans == 0, 0], Climate[y_kmeans == 0, 1], 
            s = 50, c = 'blue',label = 'cluster 0')
plt.scatter(Climate[y_kmeans == 1, 0], Climate[y_kmeans == 1, 1], 
            s = 50, c = 'purple',label = 'cluster 1')
plt.scatter(Climate[y_kmeans == 2, 0], Climate[y_kmeans == 2, 1], 
            s = 50, c = 'yellow',label = 'cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 30, c = 'red', label = 'Centroids')
plt.title('K-means Clustering of the Breast Cancer Dataset')
plt.xlabel('1980')
plt.ylabel('2019')
plt.legend()
plt.show()

# Compute the silhouette score
score = silhouette_score(Climate, y_kmeans)
print(f"Silhouette score(n=3): {score:.3f}")

