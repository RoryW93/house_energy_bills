"""
Initial analysis of meteorological/weather data sourced from the met office:
https://www.metoffice.gov.uk/research/climate/maps-and-data/historic-station-data

The website details lists all historic stations

By default, Sutton Bonington weather data is loaded in

Historic station data consists of:
- Mean daily maximum temperature (tmax)
- Mean daily minimum temperature (tmin)
- Days of air frost (af)
- Total rainfall (rain)
- Total sunshine duration (sun)

Meteorological/weat data processing and analysis is summarised as follows:

(1) Load in weather data
- create a local weather data store and load in?
- check if current txt up-to-date or needs downloaded version?

(2) Parse weather data
- update parser to deal with "provisional dataset"

(3) Visualise raw weather datasets
- time-series
- fourier transforms/spectra

(4) Visualise weather data columns distributions
- calculate box & whisker plot
- calculate histograms
- evaluate rough distribution (PDF) assuming gaussian distribution
- calculate mean and variance

(5) Compute correlations of weather data columns
- confusion matrix
- 2D and 3D visualisations of time-series data plotted against each other (include pearson correlations)

(6) Compute mean cycles of each weather data column
- pull out key features for each month: mean +/ s.d (variance)
- evaluate largest variance across each month
- determine trend lines for each time-series across a fixed month with time
- visualise linear trend line gradient/coefficient for fixed month with time as a function of month

(7) Perform PCA - data dimensionality analysis

(8) Evaluate clusters/sub-structures within data

(9) Any other exploratory data analysis or unsupervised learning?
- Fit a transformer?

(10) Modelling/Forecasting?
- evaluate goodness of fit?
- regression models?
- time-series ARIMA/SARIMA?
- LSTM?
- classification model?

Date: 04/11/2023
Author: Dr Rory White
Location: Nottingham, UK

Original functionality for class methods written: 29/10/2023

V2 - Significant update of class methods, R.White 19/07/25
- updated comments
- frequency analysis functionality added and used in: visualise_raw_weather_data
- additional plots of processed weather outputted for:
"""

import numpy as np
import pandas as pd
import regex as re
from numpy.fft import fft, fftfreq, ifft
from numpy.linalg import eig
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import norm
from scipy import signal
from scipy import interpolate
from scipy import io
from datetime import datetime, date

import os
import sys
import socket
import requests


class WeatherData:
    pass

    # Constructor
    def __init__(self, filename):  # or *argv for a variable number of arguments

        # check if filename is passed through or as a string variable
        if not filename or filename is None:
            self.filename = \
                'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/stationdata/suttonboningtondata.txt'
        elif isinstance(filename, str):
            self.filename = filename

        # check if input filename is locally stored
        self.check_input_file(filename)

        self.Wdata = None  # parsed & initial processed weather data
        self.Wdata_ff = pd.DataFrame()  # frequency-amplitude response spectrum weather data
        self.today = date.today().strftime("%d/%m/%Y")  # today's date as a string

        self.Wdata_fnat = None  # weather data cyclical fundamental frequencies
        self.data_labels = np.array([['tmin', '($^\circ$C)'],
                                     ['tmax', '($^\circ$C)'],
                                     ['tavg', '($^\circ$C)'],
                                     ['af', 'days'],
                                     ['rain', 'mm'],
                                     ['sun', 'hrs']
                                     ])
        self.weather_station = None  # weather station metadata

        self.Tmin = None  # post-processed annual Tmin data
        self.Tmax = None  # post-processed annual Tmax data
        self.Tavg = None  # post-processed annual Tavg data
        self.rain = None  # post-processed annual rain data
        self.sun = None  # post-processed annual sun data

        self.Tmin_cycle = pd.DataFrame()  #
        self.Tmax_cycle = pd.DataFrame()
        self.Tavg_cycle = pd.DataFrame()
        self.rain_cycle = pd.DataFrame()
        self.sun_cycle = pd.DataFrame()

    def check_input_file(self, filename):

        # parse input filename to extract filename -
        #'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/stationdata/suttonboningtondata.txt'#

        parts = filename.split('/')
        input_file = parts[-1]

        # Extract weather station name/location
        input_file_parts = input_file.split('.')
        weather_station = input_file_parts[0].replace('data', '')
        self.weather_station = weather_station

        # create input filename query
        query_filename = os.path.join(os.getcwd() + '..\\Input data\\' + input_file)

        if not os.path.isfile(query_filename):

            print(f'\nFile {input_file} does not exist. Downloading now....')

            # might need updating to be the url up to:
            # 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/stationdata/
            # with suttonboningtondata.txt' seperatly called and laoded

            query_parameters = {"downloadformat": "txt"}
            response = requests.get(filename, params=query_parameters)

            if response.ok or response.status_code == 200:

                with open(query_filename, mode="wb") as file:
                    file.write(response.content)

            else:
                print(f'\nUnable to download {input_file} from {input_file}. Check the ')
                exit()

    @staticmethod
    def is_float(string):
        """
        Determines if a float is contained within a string vector

        :param string: Input string vector
        :return:
        """
        try:
            float(string)
            return True
        except ValueError:
            return False

    def parse_weather_data(self, plotopt):
        """
        Parses the specific met office meteorological data format into a processed and usable pandas dataframe
        :param plotopt: Variable to switch regex parsing distribution
        :return: data - parsed, pre-processed weather data
        """

        def parse_data_row(components, counter, ii):
            """
            Helper function to help parse rows of data
            :param components:
            :param counter:
            :param ii:
            :return:
            """

            # remove empty and whitespace strings from list
            components = [x for x in components if x.strip()]
            # unpack components into parsed data variables
            for jj in range(len(components)):
                if components[jj] == "---":
                    data.iloc[counter, jj] = np.nan
                elif components[jj].isnumeric():
                    data.iloc[counter, jj] = int(components[jj])
                elif self.is_float(components[jj]):
                    data.iloc[counter, jj] = float(components[jj])

            Ccounter[counter, ii] = 1
            print(str(counter) + ' - a')
            print(components)

        # load in the raw data
        weather = pd.read_csv(self.filename, header=None, skiprows=5)  # , delimiter=r"\s+")

        # unpack header and define columns for parsed data
        columns = [weather[0][0][3:7], weather[0][0][9:11], weather[0][0][14:18], weather[0][0][22:26],
                   weather[0][0][32:34], weather[0][0][38:42], weather[0][0][47:50]]

        # define data array to contain parsed data
        data = np.zeros((weather.size - 2, len(columns)))
        data = pd.DataFrame(data, columns=columns)

        # initialise counter for parsing iterations & analysis
        counter = 0

        # 26 letters in alphabet - encode different regex configurations
        # [a,b,c,d,e,f,g,h,i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z]
        # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        Ccounter = np.empty((len(weather[0]) - 2, len(alphabet)))

        # parse weather data
        for row in weather[0]:

            # catch header data - first row
            if re.search(r"\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)", row):
                counter += 1
                components = re.split(r"\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)", row)
                print(counter)
                print(components)

            # catch header data - second row
            elif re.search(r"\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)", row):
                counter += 1
                components = re.split(r"\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)", row)
                print(counter)
                print(components)

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d)\s+(\d+)\s+(\d+.\d)\s+(---)\s+\w+", row):
                ii = 0  # a
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d)\s+(\d+)\s+(\d+.\d)\s+(---)\s+\w+",
                                      row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(str(counter) + ' - a')
                print(components)

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d)\s+(\d+)\s+(\d+.\d)\s+(---)\s+\w+", row):
                ii = 1  # b
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d)\s+(\d+)\s+(\d+.\d)\s+(---)\s+\w+",
                                      row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(str(counter) + ' - b')
                print(components)

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d)\s+(\d+.\d)\s+(\d+)\s+(---)\s+(\d+.\d)", row):
                ii = 2  # c
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d)\s+(\d+.\d)\s+(\d+)\s+(---)\s+(\d+.\d)", row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(str(counter) + ' - c')
                print(components)

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d)\s+(-\d+.\d)\s+(\d+)\s+(---)\s+(\d+.\d)", row):
                ii = 3  # d

                # determine first data row entry after header rows parsed
                # re-start counter to zero for first data row entry
                if counter == 2:
                    counter = 0
                else:
                    counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d)\s+(-\d+.\d)\s+(\d+)\s+(---)\s+(\d+.\d)", row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(str(counter) + ' - d')
                print(components)

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d)\s+(-\d+.\d)\s+(\d+)\s+(---)\s+(\d+.\d)", row):
                ii = 4  # e
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d)\s+(-\d+.\d)\s+(\d+)\s+(---)\s+(\d+.\d)", row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - e')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d)\s+(-\d+.\d)\s+(\d+)\s+(\d+.\d)\s+(\d+.\d)", row):
                ii = 5  # f
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d)\s+(-\d+.\d)\s+(\d+)\s+(\d+.\d)\s+(\d+.\d)", row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - f')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d)\s+(\d+.\d)\s+(\d+)\s+(\d+.\d)\s+(\d+.\d)", row):
                ii = 6  # g
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d)\s+(\d+.\d)\s+(\d+)\s+(\d+.\d)\s+(\d+.\d)", row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - g')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)[*]\s+(-\d+.\d)\s+(\d+)\s+(\d+.\d)\s+(\d+.\d)", row):
                ii = 7  # h
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)[*]\s+(-\d+.\d)\s+(\d+)\s+(\d+.\d)\s+(\d+.\d)",
                                      row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - h')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)[*]\s+(\d+.\d)\s+(\d+)\s+(\d+.\d)\s+(\d+.\d)", row):
                ii = 8  # i
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)[*]\s+(\d+.\d)\s+(\d+)\s+(\d+.\d)\s+(\d+.\d)",
                                      row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - i')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d)\s+(\d+.\d)\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)[*]", row):
                ii = 9  # j
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d)\s+(\d+.\d)\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)[*]",
                                      row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - j')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+)\s+(\d+.\d+)\s+(---)", row):
                ii = 10  # k
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+)\s+(\d+.\d+)\s+(---)", row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - k')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)[*]\s+(\d+)[*]\s+(\d+.\d+)\s+(\d+.\d+)", row):
                ii = 11  # l
                counter += 1

                components = re.split(
                    r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)[*]\s+(\d+)[*]\s+(\d+.\d+)\s+(\d+.\d+)",
                    row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - l')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+)\s+(\d+.\d+)[*]\s+(\d+.\d+)", row):
                ii = 12  # m
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+)\s+(\d+.\d+)[*]\s+(\d+.\d+)",
                                      row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - m')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+)\s+(\d+.\d+)\s+(---)", row):
                ii = 13  # n
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+)\s+(\d+.\d+)\s+(---)", row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - n')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(-\d+.\d+)\s+(\d+)\s+(\d+.\d+)\s+(---)", row):
                ii = 14  # o
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(-\d+.\d+)\s+(\d+)\s+(\d+.\d+)\s+(---)", row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - o')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)[*]\s+(\d+.\d+)[*]\s+(\d+)[*]\s+(\d+.\d+)\s+(---)", row):
                ii = 15  # p
                counter += 1

                components = re.split(
                    r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)[*]\s+(\d+.\d+)[*]\s+(\d+)[*]\s+(\d+.\d+)\s+(---)",
                    row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - p')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)[*]\s+(\d+.\d+)[*]\s+(\d+)[*]\s+(\d+.\d+)\s+(---)", row):
                ii = 16  # q
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+)\s+(\d+.\d+)[*]\s+(---)", row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - q')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)[*]\s+(\d+.\d+)[*]\s+(\d+)[*]\s+(\d+.\d+)\s+(---)", row):
                ii = 17  # r
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+)\s+(\d+.\d+)[*]\s+(---)", row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - r')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)[*]\s+(\d+.\d+)[*]\s+(\d+)[*]\s+(\d+.\d+)[*]\s+(---)", row):
                ii = 18  # s
                counter += 1

                components = re.split(
                    r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)[*]\s+(\d+.\d+)[*]\s+(\d+)[*]\s+(\d+.\d+)[*]\s+(---)",
                    row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - s')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)[*]\s+(\d+.\d+)[*]\s+(\d+)[*]\s+(\d+.\d+)[*]\s+(---)", row):
                ii = 19  # t
                counter += 1

                components = re.split(
                    r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)[*]\s+(\d+.\d+)[*]\s+(\d+)[*]\s+(\d+.\d+)[*]\s+(---)",
                    row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - t')

            elif re.search(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+)\s+(\d+.\d+)[*]\s+(---)", row):
                ii = 20  # u
                counter += 1

                components = re.split(r"\s+(\d{4})\s+(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+)\s+(\d+.\d+)[*]\s+(---)", row)
                # remove empty and whitespace strings from list
                components = [x for x in components if x.strip()]
                # unpack components into parsed data variables
                for jj in range(len(components)):
                    if components[jj] == "---":
                        data.iloc[counter, jj] = np.nan
                    elif components[jj].isnumeric():
                        data.iloc[counter, jj] = int(components[jj])
                    elif self.is_float(components[jj]):
                        data.iloc[counter, jj] = float(components[jj])

                Ccounter[counter, ii] = 1
                print(components)
                print(str(counter) + ' - u')

            # catch rows of data that aren't captured by the regex statements
            else:
                print('\n')
                print(row)
                print('\n')

        # check if all data is successfully parsed - counter iterations relative to
        if counter + 1 == len(weather[0]) - 2:
            print("Data has been parsed successfully")
            print(counter + 1)
            print(len(weather[0]) - 2)
        else:
            print("Something went wrong. Not all data was parsed")
            print(counter + 1)
            print(len(weather[0]) - 2)

        # evaluate & visualise distribution of regex configurations
        counts = np.sum(Ccounter, axis=0)
        print('\n')
        print(alphabet)
        print(counts)

        if plotopt == 1:
            plt.figure(1)
            plt.bar(alphabet, counts)
            plt.ylabel('Count')
            plt.title('Distribution of regex parsing configurations')
            plt.show()

        # post-process parsed weather data
        data['yyyy'] = data['yyyy'].astype(int).astype(str)
        data['mm'] = data['mm'].astype(int).astype(str)
        data['date'] = pd.to_datetime(data[['yyyy', 'mm']].agg('-'.join, axis=1))
        data['tavg'] = (data['tmax'] + data['tmin']) / 2
        data = data.rename(columns={"yyyy": "year", "mm": "month"})
        data = data[['date', 'year', 'month', 'tmax', 'tmin', 'tavg', 'af', 'rain', 'sun']]

        self.Wdata = data

        return data

    @staticmethod
    def compute_fft(x, fs, unit, plotopt, x_limit):
        """
        This algorithm computes the double-sided fourier transform spectrum of an input signal comprising a combination of
        sines and cosines. If required, the figure of the signal's frequency-energy spectrum is constructed.

        Date: 24/03/20
        Author: Rory White
        Location: Bristol, UK

        :param x: input vector
        :param fs: size of input vector x
        :param unit: this is the unit measure of the input signal, e.g [mm]
        :param plotopt: this is the option variable
        :param x_limit: creates the x axis frequency limit for the output fft figure
        :return:
                freq_vector - positive frequency of fft calculation, measured in Hz
                fft_data - double-sided fourier spectrum excluding complex conjugates
                f_nat    - natural frequency of 1st harmonic, measured in Hz
        """
        #

        n = np.size(x)  # number of points in input vector/series x
        T = 1 / fs  # sampling period

        freqs = fftfreq(n)
        mask = freqs > 0  # masks negative frequencies

        fft_vals = fft(x)  # fft calculation
        fft_theo = 2.0 * np.abs(fft_vals / n)  # computes two-sided fourier spectrum

        # remove negative frequencies and associated magnitude values
        # freq_vector = np.linspace(0.0, 1.0/(2.0 * T), int(n / 2))   # defining frequency vector
        freq_vector = freqs[mask] * 100
        fft_data = fft_theo[mask]

        # find approximate natural frequency from max fft magnitude value
        f_nat = round(freq_vector[np.argmax(fft_data, axis=0)], 2)

        # computes the figure of the fourier spectrum if plotopt = 1
        if plotopt == 1:
            plt.figure()
            plt.plot(freq_vector, fft_data, 'k')
            # plt.semilogy(freq_vector, fft_data, 'k')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Fourier amplitude [%s/Hz]' % unit)
            plt.title('Natural frequency: %s Hz' % f_nat)

            # set the figure limits
            plt.xlim(0, x_limit)
            plt.ylim((0, 1.1 * np.max(fft_data)))

            # define font size and name of figure
            font = {'family': 'times',
                    'weight': 'normal',
                    'size': 12}

            matplotlib.rc('font', **font)

            # Add natural frequency and harmonics to plot
            plt.text(f_nat * 0.8, 1.01 * np.max(fft_data), '$f_{1} = %s $Hz' % f_nat, fontdict={'family': 'times',
                                                                                                'weight': 'normal',
                                                                                                'size': 8})

            plt.show()

        elif plotopt == 2:
            plt.figure()
            plt.semilogy(freq_vector, fft_data, 'k')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Fourier amplitude [%s/Hz]' % unit)
            plt.title('Natural frequency: %s Hz' % f_nat)

            # set the figure limits
            plt.xlim(0, x_limit)
            plt.ylim((0, 1.1 * np.max(fft_data)))

            # define font size and name of figure
            font = {'family': 'times',
                    'weight': 'normal',
                    'size': 12}

            matplotlib.rc('font', **font)

            # Add natural frequency and harmonics to plot
            plt.text(f_nat * 0.8, 1.01 * np.max(fft_data), '$f_{1} = %s $Hz' % f_nat, fontdict={'family': 'times',
                                                                                                'weight': 'normal',
                                                                                                'size': 8})
            plt.show()

        return freq_vector, fft_data, f_nat

    def visualise_raw_weather_data(self):
        """
        Visualises the raw processed weather time-series data
        Temperatures: tmin, tmax, tavg - degC
        af - days (days of air frost)
        rain - mm
        sun - hrs

        Evaluates FFT - frequency spectra of cyclical time-series data
        :return:
        """

        # -------------------------------------------------------------------------------------------------#
        # compute FFTs of signals
        # -------------------------------------------------------------------------------------------------#
        # initialise and define data variables
        self.Wdata_fnat = np.array([])

        fs = 1 / (365 * 24 * 60 ** 2 / 12)  # sampling frequency - 12 data points a year, 1/12
        plotopt = 0
        x_limit = 2

        for label in self.data_labels:
            self.Wdata_ff['freq_vector'], self.Wdata_ff[label[0]], f_nat = self.compute_fft(self.Wdata[label[0]], fs,
                                                                                            label[1], plotopt, x_limit
                                                                                            )
            self.Wdata_fnat = np.append(self.Wdata_fnat, f_nat)

        # -------------------------------------------------------------------------------------------------#
        # temperature min
        # -------------------------------------------------------------------------------------------------#
        plt.figure(1, figsize=(10, 15))
        plt.subplot(1, 2, 1)
        plt.plot(self.Wdata['date'], self.Wdata['tmin'], 'k')
        plt.xlabel('Time [yyyy-mm-dd]')
        plt.ylabel('Min. Temp [$^\circ$C]')

        plt.subplot(1, 2, 2)
        plt.plot(self.Wdata_ff['freq_vector'], self.Wdata_ff['tmin'], 'k')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Min. Temp [$^\circ$C/Hz]')
        plt.xlim([0, 25])

        # -------------------------------------------------------------------------------------------------#
        # temperature max
        # -------------------------------------------------------------------------------------------------#
        plt.figure(2, figsize=(10, 15))
        plt.subplot(1, 2, 1)
        plt.plot(self.Wdata['date'], self.Wdata['tmax'], 'k')
        plt.xlabel('Time [yyyy-mm-dd]')
        plt.ylabel('Max Temp [$^\circ$C]')

        plt.subplot(1, 2, 2)
        plt.plot(self.Wdata_ff['freq_vector'], self.Wdata_ff['tmax'], 'k')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Max Temp [$^\circ$C/Hz]')
        plt.xlim([0, 25])

        # -------------------------------------------------------------------------------------------------#
        # temperature average
        # -------------------------------------------------------------------------------------------------#
        plt.figure(3, figsize=(10, 15))
        plt.subplot(1, 2, 1)
        plt.plot(self.Wdata['date'], self.Wdata['tavg'], 'k')
        plt.xlabel('Time [yyyy-mm-dd]')
        plt.ylabel('Avg Temp [$^\circ$C]')

        plt.subplot(1, 2, 2)
        plt.plot(self.Wdata_ff['freq_vector'], self.Wdata_ff['tavg'], 'k')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Avg Temp [$^\circ$C/Hz]')
        plt.xlim([0, 25])

        # -------------------------------------------------------------------------------------------------#
        # af - days of air frost
        # -------------------------------------------------------------------------------------------------#
        plt.figure(4, figsize=(10, 15))
        plt.subplot(1, 2, 1)
        plt.plot(self.Wdata['date'], self.Wdata['af'], 'k')
        plt.xlabel('Time [yyyy-mm-dd]')
        plt.ylabel('af [days]')

        plt.subplot(1, 2, 2)
        plt.plot(self.Wdata_ff['freq_vector'], self.Wdata_ff['af'], 'k')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('af [days/Hz]')
        plt.xlim([0, 25])

        # -------------------------------------------------------------------------------------------------#
        # rain - mm
        # -------------------------------------------------------------------------------------------------#
        plt.figure(5, figsize=(10, 15))
        plt.subplot(1, 2, 1)
        plt.plot(self.Wdata['date'], self.Wdata['rain'], 'k')
        plt.xlabel('Time [yyyy-mm-dd]')
        plt.ylabel('rain [mm]')

        plt.subplot(1, 2, 2)
        plt.plot(self.Wdata_ff['freq_vector'], self.Wdata_ff['rain'], 'k')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('rain [mm/Hz]')
        plt.xlim([0, 25])

        # -------------------------------------------------------------------------------------------------#
        # sun - hrs
        # -------------------------------------------------------------------------------------------------#
        plt.figure(6, figsize=(10, 15))
        plt.subplot(1, 2, 1)
        plt.plot(self.Wdata['date'], self.Wdata['sun'], 'k')
        plt.xlabel('Time [yyyy-mm-dd]')
        plt.ylabel('sun [hrs]')
        plt.show()

        plt.subplot(1, 2, 2)
        plt.plot(self.Wdata_ff['freq_vector'], self.Wdata_ff['sun'], 'k')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('sun [hrs/Hz]')
        plt.xlim([0, 25])

    def visualise_weather_data_distributions(self):
        """
        Visualises the data distributions of the average temperature, rain and hours of sun
        - As a box and whisker plot
        - As a histogram
        :return:
        """
        # number of bins for histograms
        u_bins = int(round(np.sqrt(len(self.Wdata['tavg'])), 0))
        # u_bins = 11

        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.boxplot(self.Wdata['tavg'])
        plt.ylabel("Temp. [$^\circ$C]")
        plt.subplot(1, 2, 2)
        plt.hist(self.Wdata['tavg'], bins=u_bins)
        plt.xlabel("Temp. [$^\circ$C]")
        plt.ylabel("Count")

        plt.figure(2)
        plt.subplot(1, 2, 1)
        plt.boxplot(self.Wdata['rain'])
        plt.ylabel("rain [mm]")
        plt.subplot(1, 2, 2)
        plt.hist(self.Wdata['rain'], bins=u_bins)
        plt.xlabel("rain [mm]")
        plt.ylabel("Count")

        plt.figure(3)
        plt.subplot(1, 2, 1)
        plt.boxplot(self.Wdata['af'])
        plt.ylabel("af [days]")
        plt.subplot(1, 2, 2)
        plt.hist(self.Wdata['af'], bins=u_bins)
        plt.xlabel("af [days]")
        plt.ylabel("Count")

        plt.figure(4)
        plt.subplot(1, 2, 1)
        plt.boxplot(self.Wdata['sun'])
        plt.ylabel("sun [hrs]")
        plt.subplot(1, 2, 2)
        plt.hist(self.Wdata['sun'], bins=u_bins)
        plt.xlabel("sun [hrs]")
        plt.ylabel("Count")
        plt.show()

    def compute_correlations(self):
        """
        Evaluates the pearson correlations of all weather data variables against each other
        :return:
        """
        # print(self.Wdata.describe())
        # compute pearson correlations within entire dataset variables
        corr = self.Wdata.corr()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, len(self.Wdata.columns), 1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(self.Wdata.columns)
        ax.set_yticklabels(self.Wdata.columns)
        plt.show()

        # visualise all weather data 2D correlations
        plt.figure(0, figsize=(20, 20))
        plt.subplot(2, 2, 1)
        plt.plot(self.Wdata['tmin'], self.Wdata['tmax'], color='k', linestyle='None', marker='x')
        plt.xlabel('Min Temp ($^\circ$C)')
        plt.ylabel('Max Temp ($^\circ$C)')

        plt.subplot(2, 2, 2)
        plt.plot(self.Wdata['af'], self.Wdata['sun'], color='k', linestyle='None', marker='x')
        plt.xlabel('af [days]')
        plt.ylabel('sun [hrs]')

        plt.subplot(2, 2, 3)
        plt.plot(self.Wdata['af'], self.Wdata['rain'], color='k', linestyle='None', marker='x')
        plt.xlabel('af [days]')
        plt.ylabel('rain [mm]')

        plt.subplot(2, 2, 4)
        plt.plot(self.Wdata['rain'], self.Wdata['sun'], color='k', linestyle='None', marker='x')
        plt.xlabel('rain [mm]')
        plt.ylabel('sun [hrs]')
        plt.show()

        plt.figure(0, figsize=(20, 20))
        plt.subplot(3, 1, 1)
        plt.plot(self.Wdata['tmin'], self.Wdata['af'], color='b', linestyle='None', marker='x', label='Tmin')
        plt.plot(self.Wdata['tmax'], self.Wdata['af'], color='r', linestyle='None', marker='x', label='Tmax')
        plt.xlabel('Temp ($^\circ$C)')
        plt.ylabel('af [days]')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(self.Wdata['tmin'], self.Wdata['rain'], color='b', linestyle='None', marker='x', label='Tmin')
        plt.plot(self.Wdata['tmax'], self.Wdata['rain'], color='r', linestyle='None', marker='x', label='Tmax')
        plt.xlabel('Temp ($^\circ$C)')
        plt.ylabel('rain [mm]')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(self.Wdata['tmin'], self.Wdata['sun'], color='b', linestyle='None', marker='x', label='Tmin')
        plt.plot(self.Wdata['tmax'], self.Wdata['sun'], color='r', linestyle='None', marker='x', label='Tmax')
        plt.xlabel('Temp ($^\circ$C)')
        plt.ylabel('sun [hrs]')
        plt.legend()
        plt.show()

    def compute_mean_cycle(self):
        """
        Evaluates the mean annual cycle for each column of weather data that displays cyclical responses:
        - Tmin (degC)
        - Tmax (degC)
        - Tavg (degC)
        - rain (mm)
        - sun (hrs)

        Includes the annual cycle variance associated with the mean calculated.
        Following plots generated:
        (1) - Visualises annual weather data cycle distributions
        (2) - Visualises weather data monthly variation over the years
        :return:
        """

        # convert string columns datasets into integer type
        self.Wdata['year'] = self.Wdata['year'].astype(int)
        self.Wdata['month'] = self.Wdata['month'].astype(int)

        # 12 data points per year - define frequency vector for post-processing - dppy (data points per year)
        frequency = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        frequency_new = frequency

        # year samples
        samples = len(self.Wdata) / 12
        samples_int = int(round(samples, 0))
        samples_float = samples - samples_int

        # if samples_float is +ve - fraction above samples_int calculated
        if samples_float > 0:
            samples_diff = round(12 * samples_float, 0)

        # if samples_float is -ve - fraction below samples_int calculated
        elif samples_float < 0:
            samples_diff = round(12 * samples_float, 0) + 12

        # if samples_float is 0
        elif samples_float == 0:
            samples_diff = 0

        # initialise vectors
        Tmin = np.empty((len(frequency), samples_int))  # Wdata['tmin']
        Tmax = np.empty((len(frequency), samples_int))  # Wdata['tmax']
        Tavg = np.empty((len(frequency), samples_int))  # Wdata['tavg']
        rain = np.empty((len(frequency), samples_int))  # Wdata['rain']
        sun = np.empty((len(frequency), samples_int))  # Wdata['sun']

        # iterate and pack parsed weather data per annum - 12 data points per year
        for ii in range(samples_int):

            # determine if final year of data is a full year or not
            if (ii == samples_int - 2) and (0 < np.abs(samples_float) < 1):
                Tmin[frequency, ii] = self.Wdata['tmin'][frequency_new]
                Tavg[frequency, ii] = self.Wdata['tavg'][frequency_new]
                Tmax[frequency, ii] = self.Wdata['tmax'][frequency_new]
                rain[frequency, ii] = self.Wdata['rain'][frequency_new]
                sun[frequency, ii] = self.Wdata['sun'][frequency_new]

                frequency_new = np.linspace(frequency_new[-1] + 1,
                                            int(frequency_new[-1] + samples_diff),
                                            int(samples_diff))

            # final column iteration
            elif (ii == samples_int - 1) and (0 < np.abs(samples_float) < 1):
                Tmin[frequency[0:int(samples_diff)], ii] = self.Wdata['tmin'][frequency_new]
                Tavg[frequency[0:int(samples_diff)], ii] = self.Wdata['tavg'][frequency_new]
                Tmax[frequency[0:int(samples_diff)], ii] = self.Wdata['tmax'][frequency_new]
                rain[frequency[0:int(samples_diff)], ii] = self.Wdata['rain'][frequency_new]
                sun[frequency[0:int(samples_diff)], ii] = self.Wdata['sun'][frequency_new]

                # handle missing data points for rest of year - samples difference with NaN handling
                Tmin[[-1 + int(samples_diff) - 12 + 1, -1], ii] = np.nan
                Tavg[[-1 + int(samples_diff) - 12 + 1, -1], ii] = np.nan
                Tmax[[-1 + int(samples_diff) - 12 + 1, -1], ii] = np.nan
                rain[[-1 + int(samples_diff) - 12 + 1, -1], ii] = np.nan
                sun[[-1 + int(samples_diff) - 12 + 1, -1], ii] = np.nan

            # first to third to last iterations
            else:
                Tmin[frequency, ii] = self.Wdata['tmin'][frequency_new]
                Tavg[frequency, ii] = self.Wdata['tavg'][frequency_new]
                Tmax[frequency, ii] = self.Wdata['tmax'][frequency_new]
                rain[frequency, ii] = self.Wdata['rain'][frequency_new]
                sun[frequency, ii] = self.Wdata['sun'][frequency_new]

                frequency_new = frequency + frequency_new[-1] + 1

        # create mean cycles
        Tmin_mean = np.nanmean(Tmin, axis=1)  # .reshape(12,1)
        Tmax_mean = np.nanmean(Tmax, axis=1)  # .reshape(12,1)
        Tavg_mean = np.nanmean(Tavg, axis=1)  # .reshape(12,1)
        rain_mean = np.nanmean(rain, axis=1)  # .reshape(12,1)
        sun_mean = np.nanmean(sun, axis=1)

        # create std variance for overall mean cycle
        Tmin_std = np.nanstd(Tmin, axis=1)  # .reshape(12,1)
        Tmax_std = np.nanstd(Tmax, axis=1)  # .reshape(12,1)
        Tavg_std = np.nanstd(Tavg, axis=1)  # .reshape(12,1)
        rain_std = np.nanstd(rain, axis=1)  # .reshape(12,1)
        sun_std = np.nanstd(sun, axis=1)

        # Pack post-processed data variables into individual pandas dataframes
        # Note: .reshape(12,1) not required
        self.Tmin = Tmin
        self.Tmin_cycle['mean'] = Tmin_mean
        self.Tmin_cycle['std'] = Tmin_std

        self.Tmax = Tmax
        self.Tmax_cycle['mean'] = Tmax_mean
        self.Tmax_cycle['std'] = Tmax_std

        self.Tavg = Tavg
        self.Tavg_cycle['mean'] = Tavg_mean
        self.Tavg_cycle['std'] = Tavg_std

        self.rain = rain
        self.rain_cycle['mean'] = rain_mean
        self.rain_cycle['std'] = rain_std

        self.sun = sun
        self.sun_cycle['mean'] = sun_mean
        self.sun_cycle['std'] = sun_std

        # create figure of annual data cycle distributions: mean & standard deviation
        plt.figure(1)
        plt.plot(frequency + 1, Tmin_mean, 'r', linestyle='-', linewidth=2, label='mean')
        plt.plot(frequency + 1, Tmin_mean + Tmin_std, 'k', linestyle='--', linewidth=2, label='mean+/-std')
        plt.plot(frequency + 1, Tmin_mean - Tmin_std, 'k', linestyle='--', linewidth=2)
        plt.xlabel('Month')
        plt.ylabel('Tmin [degC]')
        plt.legend()
        for ii in range(samples_int):
            plt.plot(frequency + 1, Tmin[:, ii], 'b', linestyle='-', linewidth=0.1)

        plt.figure(2)
        plt.plot(frequency + 1, Tmax_mean, 'r', linestyle='-', linewidth=2, label='mean')
        plt.plot(frequency + 1, Tmax_mean + Tmax_std, 'k', linestyle='--', linewidth=2, label='mean+/-std')
        plt.plot(frequency + 1, Tmax_mean - Tmax_std, 'k', linestyle='--', linewidth=2)
        plt.xlabel('Month')
        plt.ylabel('Tmax [degC]')
        plt.legend()
        for ii in range(samples_int):
            plt.plot(frequency + 1, Tmax[:, ii], 'b', linestyle='-', linewidth=0.1)

        plt.figure(3)
        plt.plot(frequency + 1, Tavg_mean, 'r', linestyle='-', linewidth=2, label='mean')
        plt.plot(frequency + 1, Tavg_mean + Tavg_std, 'k', linestyle='--', linewidth=2, label='mean+/-std')
        plt.plot(frequency + 1, Tavg_mean - Tavg_std, 'k', linestyle='--', linewidth=2)
        plt.xlabel('Month')
        plt.ylabel('Tavg [degC]')
        plt.legend()
        for ii in range(samples_int):
            plt.plot(frequency + 1, Tavg[:, ii], 'b', linestyle='-', linewidth=0.1)

        plt.figure(4)
        plt.plot(frequency + 1, sun_mean, 'r', linestyle='-', linewidth=2, label='mean')
        plt.plot(frequency + 1, sun_mean + sun_std, 'k', linestyle='--', linewidth=2, label='mean+/-std')
        plt.plot(frequency + 1, sun_mean - sun_std, 'k', linestyle='--', linewidth=2)
        plt.xlabel('Month')
        plt.ylabel('sun [hrs]')
        plt.legend()
        for ii in range(samples_int):
            plt.plot(frequency + 1, sun[:, ii], 'b', linestyle='-', linewidth=0.1)
        plt.show()

        # Plot monthly time-histories over each year to identify temperature trends
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                  'August', 'September', 'October', 'November', 'December']
        for ii in range(12):
            # Temperature
            plt.figure(5 + ii)
            plt.plot(Tmax[ii, :], 'r-', label='max')
            plt.plot(Tavg[ii, :], 'k-', label='avg')
            plt.plot(Tmin[ii, :], 'b-', label='min')
            plt.xlabel('Year')
            plt.ylabel(months[ii] + ' temperature [degC]')
            plt.ylim([-5, 35])
            plt.legend()

            # Rain
            plt.figure(50 + ii)
            plt.plot(rain[ii, :], 'k-')
            plt.xlabel('Year')
            plt.ylabel(months[ii] + ' rain [mm]')
            # plt.ylim([-5, 35])

            # Sun
            plt.figure(500 + ii)
            plt.plot(sun[ii, :], 'k-')
            plt.xlabel('Year')
            plt.ylabel(months[ii] + ' sun [hrs]')
            # plt.ylim([-5, 35])
            plt.show()
            plt.close('all')

    def process_cyclic_data(self):

        def calc_Amp(x):
            return (np.max(x, axis=1) - np.min(x, axis=1))/2

        # calculate cycle-by-cycle frequency and amplitudes
        Tmax_A = calc_Amp(self.Tmax)
        Tavg_A = calc_Amp(self.Tavg)
        Tmin_A = calc_Amp(self.Tmin)

        # calculate total average annual rain per year - trpy
        trpy = np.sum(self.rain, axis=1)
        tspy = np.sum(self.sun, axis=1)

        print(f' Max temperature amplitude: {np.round(np.mean(Tmax_A), 2)} \u00B1 {np.round(np.std(Tmax_A), 2)} \n')

        # number of bins for histograms
        u_bins = int(round(np.sqrt(len(Tmax_A)), 0))
        # u_bins = 11

        # evaluate variations
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.boxplot(Tmax_A)
        plt.ylabel("Temp. [$^\circ$C]")
        plt.subplot(1, 2, 2)
        plt.hist(Tmax_A, bins=u_bins)
        plt.xlabel("Temp. [$^\circ$C]")
        plt.ylabel("Count")

        plt.figure(2)
        plt.subplot(1, 2, 1)
        plt.boxplot(Tavg_A)
        plt.ylabel("Temp. [$^\circ$C]")
        plt.subplot(1, 2, 2)
        plt.hist(Tavg_A, bins=u_bins)
        plt.xlabel("Temp. [$^\circ$C]")
        plt.ylabel("Count")

        plt.figure(3)
        plt.subplot(1, 2, 1)
        plt.boxplot(Tmin_A)
        plt.ylabel("Temp. [$^\circ$C]")
        plt.subplot(1, 2, 2)
        plt.hist(Tmin_A, bins=u_bins)
        plt.xlabel("Temp. [$^\circ$C]")
        plt.ylabel("Count")

        plt.figure(4)
        plt.subplot(1, 2, 1)
        plt.boxplot(trpy)
        plt.ylabel("Total rain per year [mm]")
        plt.subplot(1, 2, 2)
        plt.hist(trpy, bins=u_bins)
        plt.xlabel("Total rain per year [mm]")
        plt.ylabel("Count")

        plt.figure(5)
        plt.subplot(1, 2, 1)
        plt.boxplot(tspy)
        plt.ylabel("Total rain per year [mm]")
        plt.subplot(1, 2, 2)
        plt.hist(tspy, bins=u_bins)
        plt.xlabel("Total rain per year [mm]")
        plt.ylabel("Count")
        plt.show()

    @staticmethod
    def reduce_dim(X):
        """
        Perform dimensionality reduction using principal component analysis - singular value decomposition
        :param X: Input data vector
        :return:
        """
        dofs = X.shape[1]
        pca = PCA(n_components=dofs)
        Xt = pca.fit_transform(X)

        print(pca.explained_variance_ratio_ * 100)

        pca_list = []
        for x in range(len(pca.explained_variance_ratio_)):
            # if x*100 > 1:
            pca_list.append("PC" + str(x + 1))
        # pca_columns = [x for x in range(dofs) "PC" + int(x)]

        # Visualise scree plot
        plt.figure(1)
        plt.bar(pca_list, pca.explained_variance_ratio_ * 100)
        plt.ylim([0, 100])
        plt.ylabel('Variance captured [%]')
        plt.title('Scree plot: [tmin,tmax,rain,sun]')
        plt.savefig('Figures\\weather_data_pca_dimensionality_analysis.png', dpi=800)
        plt.show()

        # rescale the dataset first
        # pca = PCA()
        # pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
        # Xt = pipe.fit_transform(X)

        return pca, Xt

    @staticmethod
    def number_of_clusters(data):
        """
        Determine the number of clusters (groups) inherent in the data based on KMeans cluster elbow technique.
        The gradient of the sum of squared error (sse) function is minimised to determine the location of significant
        reduction in error.
        :param data: Input pandas dataframe
        :return: clusters_num - number of cluster groups
        """

        # Perform Kmeans cluster elbow technique
        sse = {}  # use the sum of squared error as a quantitative error metric
        for k in range(1, 5):
            kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
            sse[k] = kmeans.inertia_

        # KMeans cluster elbow analysis
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(list(sse.keys()), list(sse.values()))
        plt.xlabel("Number of clusters")
        plt.ylabel("SSE")
        plt.title('k-means clustering elbow technique')
        plt.subplot(2, 1, 2)
        plt.plot(np.diff(list(sse.values())))
        plt.ylabel('Gradient of SSE (elbow)')
        # plt.show()
        plt.savefig('Figures\\Kmeans_cluster_analysis.png', dpi=800)

        # minimise the gradient of the sse function - find nominal number of clusters
        min_pos = list(abs(np.diff(list(sse.values()))))
        clusters_num = min_pos.index(min(min_pos)) + 1

        model_string_list = ['Data cluster (grouping) analysis:\n',
                             f'Number of clusters: {clusters_num}']
        self.log_file(model_string_list)

        return clusters_num

    @staticmethod
    def compute_cluster_model(data):
        """
        Create Kmeans cluster model using processed pandas dataframe data is clustered into groups
        based on an elbow technique analysis to determine the nominal number of clusters.

        Note: Make sure to pass in exact features from pandas dataframe

        :param data: Input pandas dataframe
        :return: labels - cluster group data labels
        """

        # determine nominal number of clusters
        clusters_num = WeatherData.number_of_clusters(data)

        # Develop model and determine cluster groups/labels
        model = KMeans(n_clusters=clusters_num)
        model.fit(data)
        labels = model.predict(data)

        self.cluster_model = model
        self.cluster_labels = labels

        return labels
