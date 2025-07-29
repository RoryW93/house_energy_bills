"""
Initial analysis of meteorological/weather data
(1) Load in weather data
(2) Parse weather data
(3) Visualise raw weather datasets
(4) Visualise weather data columns distributions
(5) Compute correlations of weather data columns
(6) Compute mean cycles of each weather data column
(7) Perform PCA - data dimensionality analysis
(8) Evaluate any cluster formations within data

Date: 29/10/2023
Author: Dr Rory White
Location: Nottingham, UK
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import regex as re
from WeatherData import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# (1) Instantiate WeatherData class
filename = 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/stationdata/suttonboningtondata.txt'
weather = WeatherData(filename)

# (2) Parse weather data into usable data format
plotopt = 0
weather.parse_weather_data(plotopt)

# (3) Visualise raw weather datasets
weather.visualise_raw_weather_data()

# (4) Visualise weather data columns distributions
weather.visualise_weather_data_distributions()

# (5) Compute correlations of weather data columns
weather.compute_correlations()

# (6) Compute mean cycles of each weather data column
weather.compute_mean_cycle()

# (7) Perform PCA - dimensionality analysis data
X = weather.Wdata[['tmin', 'tmax', 'rain', 'sun']].copy().to_numpy() # [['tmin', 'tavg', 'tmax', 'rain', 'sun']]
# pre-process data to remove NaNs and estimate their values based on multivariate imputation - estimator model
# Note thta IterativeImputer is experimental and syntax/format might change
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(X)
pca, Xt = weather.reduce_dim(imp.transform(X))

# visualise the reduced dataset
plt.figure()
plt.subplot(2,1,1)
plt.plot(weather.Wdata['date'], Xt[:,0], 'k-', label='PC1')
plt.ylabel("Reduced dataset - PC1")
plt.subplot(2,1,2)
plt.plot(weather.Wdata['date'], Xt[:,1], 'b-', label='PC2')
plt.xlabel("Time [yyyy/mm]")
plt.ylabel("Reduced dataset - PC2")
# plt.legend()
plt.savefig('Figures\\reduced_2DOF_weather_data.png', dpi=800)
plt.show()

aa = 1

# principle components - 2D
plt.figure()
plt.plot(Xt[:,0], Xt[:,1], color='b', linestyle='None', marker='o', markersize=2)
plt.xlabel("PC1")
plt.ylabel("PC2")
# plt.legend()
plt.savefig('Figures\\weather_data_PCA.png', dpi=800)
plt.show()

# principle components - 3D?
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(Xt[:,0], Xt[:,1], Xt[:,3], color='b', linestyle='None', marker='o', markersize=2)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()

# Evaluate any cluster formations within data
# Kmeans & nearest neighbour
