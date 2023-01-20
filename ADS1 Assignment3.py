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
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

# Define functions to read and transpose file
def readfile(doc,columns,indicator):
    """
    Reads data, extracts columns and indicator.
    sets index, returns data in original and 
    transposed format
    """
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

# Plot the result using scatter plot
plt.scatter(Climate[:,0], Climate[:,1], label="Countries")
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
ClimateChange['Cluster'] = kmeans.labels_ #add new column and label it cluster
print(ClimateChange)

# Compute the silhouette score
score = silhouette_score(Climate, y_kmeans)
print(f"Silhouette score(n=3): {score:.3f}")

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
plt.title('Plot showing cluster membership and cluster centres')
plt.xlabel('1980')
plt.ylabel('2019')
plt.legend()
plt.show()

# Define function to read file
def solution(doc):
    """Reads data"""
    filedata = pd.read_excel(doc, skiprows=3)
    return filedata

WorldData = solution('climate change.xls')
print(WorldData)

# Drop the columns
WorldData = WorldData.drop(['Country Code', 'Indicator Name', 
                            'Indicator Code'], axis=1)

# Transpose the original data
World_Data = WorldData.T
print(World_Data)

# Rename the columns
World_Data = World_Data.rename(columns=World_Data.iloc[0])

# Drop the country name
World_Data = World_Data.drop(index=World_Data.index[0], axis=0)
World_Data['Year'] = World_Data.index

# Extract data for country, convert to float and remove errors
data_fit = World_Data[['Year', 'Canada']].apply(pd.to_numeric, 
                                               errors='coerce')
print(data_fit)

# Drop null values and convert data to an array
World_New = data_fit.dropna(axis=1).values
print(World_New)

# Extract values for X and Y axis
x_axis = World_New[:,0]
y_axis = World_New[:,1]

# Define function for Curve Fit
def polynomial(x, a, b, c, d):
    '''
    Calculates polynomial function
    Function for fitting
    x: independent variable
    a, b, c, d: parameters to be fitted
    '''
    return a*x**3 + b*x**2 + c*x + d

# Fit the data
popt, covar = opt.curve_fit(polynomial, x_axis, y_axis)
a, b, c, d = popt
print(a, b, c, d)

# Plot the result
plt.scatter(x_axis, y_axis)
plt.title('Plot showing Country against Year')
plt.xlabel("Year")
plt.ylabel("Canada")
plt.legend()
plt.show()

x_line = np.arange(min(x_axis), max(x_axis)+1, 1)
y_line = polynomial(x_line, a, b, c, d)

# Plot the result
plt.scatter(x_axis, y_axis, label="Canada")
plt.plot(x_line, y_line, '--', color='black', label="forecast")
plt.xlabel("Year")
plt.ylabel("Canada")
plt.title('Curve Fit Plot showing Forecast superimposed on Country')
plt.legend()
plt.show()

# Estimate and print Lower and Upper Limits of Confidence Range
sigma = np.sqrt(np.diag(covar))
low, up = err.err_ranges(x_axis, polynomial, popt, sigma)

ci = 1.96*np.std(y_axis)/np.sqrt(len(x_axis))
low = y_line - ci
up = y_line + ci

# Plot the result
plt.scatter(x_axis, y_axis, label="Canada")
plt.plot(x_line, y_line, '--', color='black', label="forecast")
plt.fill_between(x_line, low, up, color="yellow", alpha=0.2)
plt.title('Plot showing Fitting Function and Confidence Range')
plt.xlabel("Year")
plt.ylabel("Canada")
plt.legend()
plt.show()
