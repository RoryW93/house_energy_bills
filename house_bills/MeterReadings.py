"""
Functionality of meter readings GUI/initial scripting

Write as class object - MeterReadings

(1) load in most recent dataset (.txt) file as a table/spreadsheet
(2) Prompt program user to input gas and electricity meter readings in (Gas in m^3 & electricity in kwh)
    Fault-check data to ensure:
        - No repeats are inputted
        - The correct data type is inputted
(3) Write new data to existing .txt file
(4) Plot/Visualise gas & electricity as a time-history
(5) Compute the difference, amount used, from month-month (datapoint-datapoint) & visualise as a time-history
(6a) Compute the overall statistics of the monthly amounts used and print out mean +- std
(6b) Generate the probability distribution function

Date:14/03/23
Author: Rory White
Location: Nottingham, UK

Updated functionality 02/08/23 - R.White

Further functionality developed 02/11/23 - R.White

Add functionality to define if cost data is present or not
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import socket
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import scipy.stats as st
import re
from datetime import datetime, date
from PIL import Image, ImageTk
from tkinter import Label
from WeatherData import *


class MeterReadings:
    pass

    # Constructor
    def __init__(self, filename, mode):  # or *argv for a variable number of arguments

        # check if filename is passed through or as a string variable
        if not filename or filename is None:
            self.filename = 'meter_readings.txt'
            self.mode = 1
        elif isinstance(filename, str) & isinstance(mode, int):
            self.filename = filename
            self.mode = mode

        self.data = None  # raw input data
        self.weather = None  # raw weather data - own seperate class
        self.bills_consumed = None  # monthly bills consumed
        self.cost = None  # monthly costs
        self.processed_data = None  # processed energy and cost datasets combined
        self.daily_usage = None  # estimated average daily usage of energy bills and cost per month
        self.pdf = None  # probability distribution functions
        self.model = None  # regression model - Gas & electricity
        self.data2model = None  # processed dataset for multi-linear regression
        self.today = date.today().strftime("%d/%m/%Y").replace('/', '-')  # today's date as a string
        # log file to capture dataset
        self.log_filename = 'Logs\\Data_analytics_' + socket.gethostname() + '_' + self.today + '_logfile.txt'
        self.cluster_model = None
        self.cluster_labels = None
        # self.pdf.elec = None
        # self.pdf.gas = None
        #
        # self.pdf.TwoD = None

        # Write header for log file
        header_string_list = ['------------------------------------------------\n',
                              '|Energy Bills Data Modelling & Analysis Log File|\n',
                              '------------------------------------------------\n',
                              f'\nDate: {self.today},'
                              f'\nHost: {socket.gethostname()}'
                              f'\nLog file: {self.log_filename}']
        self.log_file(header_string_list)

        # perform main method thread
        if self.mode == 1:
            print('Running as entire package - application')
            self.main()
        elif self.mode == 2:
            print('Running individually')

    @staticmethod
    def load_file(filename):
        """
        Loads in the meter_readings.txt file as a pandas dataframe
        Data columns are formatted as: Time [dd/mm/yyyy], Gas [m^3], Electricity [kwh]
        Loads in energy_bills.txt file as a pandas dataframe
        Data columns are formatted as: Time [dd/mm/yyyy], Cost [£]
        :return: data
        """
        data = pd.read_csv(filename)

        # if filename.split('.')[1] == 'csv' or filename.split('.')[1] == 'txt':
        #     data = pd.read_csv(filename)
        #
        # elif filename.split('.')[1] == 'xlsx':
        #     data = pd.read_excel(filename)

        # print out first 5 entries from dataset
        print(data.head(5))

        return data

    def write_file(self):
        """
        Writes new inputted data to file - updating the meter_readings.txt
        Fault checking is performed to make sure that new input data is not repeat data
        :return:
        """

        # check if current processed datafile is the same as read in data
        temp_data = pd.read_csv(self.filename)

        if temp_data[self.data.columns.values[2]].sum() == self.data[self.data.columns.values[2]].sum():
            print('Processed data is the same as original file. Not saving......')

        else:

            directory = os.getcwd()
            # or hardcode the directory to save text data file to
            # directory = "C:\Users\roryw\PycharmProjects\house_bills"
            if self.filename.split('.')[1] == 'csv' or self.filename.split('.')[1] == 'txt':
                self.data.to_csv(os.path.join(directory, self.filename), index=False)

            elif self.filename.split('.')[1] == 'xlsx':
                self.data.to_excel(directory, self.filename)

    def input_new_data(self):
        """
        Input new data as an entry within the most updated data variable
        :return:
        """

        input_string = input("Input new data in the following format, Time [dd/mm/yyyy], Gas [m^3], Electricity ["
                             "kwh]:")

        # Fault-check data to ensure the correct data type is inputted
        if not isinstance(input_string, str):
            print('Input data is in the incorrect format - input data as a string delimited by a space',
                  file=sys.stderr)
            input_string = \
                input("Input new data in the following format, Time [dd/mm/yyyy], Gas [m^3], Electricity [kwh]:")

        user_list = input_string.split(" ")
        # Fault-check data to ensure the correct data type is inputted
        if not isinstance(user_list, list):
            input_string = \
                input("Input new data in the following format, Time [dd/mm/yyyy], Gas [m^3], Electricity [kwh]:")
            user_list = input_string.split(" ")

        # convert each item to int type
        list_data = []
        for i in range(len(user_list)):

            if "'" in user_list[i]:
                user_list[i] = user_list[i].replace("'", "")
            elif '"' in user_list[i]:
                user_list[i] = user_list[i].replace('"', '')

            # convert the time data to datetime type
            if re.search(r'\d+/\d+/\d+', user_list[i]):
                list_data.append(user_list[i])
                # user_list[i], datetime.strptime(user_list[i], '%d/%m/%y')
            else:
                # convert numeric items to int type
                list_data.append(int(user_list[i]))

        # concatenate new data to existing dataset - new data as new index entry in dataframe
        # column_names = self.data.columns
        input_data = DataFrame(list_data).transpose()
        input_data.columns = self.data.columns
        self.data = pd.concat([self.data, input_data])
        self.data[self.data.columns.values[0]] = pd.to_datetime(self.data[self.data.columns.values[0]],
                                                                format="%d/%m/%Y")

        # Fault-check data to ensure: No repeats are inputted
        bool_series = self.data.duplicated(keep='first')
        if True in bool_series:
            print("New input data identified as a repeated data entry. Removing now.....")
            self.data = self.data[~bool_series]

        # Fault check data to ensure that latest entry is also larger than the previous value
        if self.data.iloc[-1, 1] < self.data.iloc[-1 + 1, 1] or self.data.iloc[-1, 2] < self.data.iloc[-1 + 1, 2]:
            print("New input data identified as smaller than previous data entry. Removing now.....")
            self.data = self.data.iloc[-1:0:-1].pop()
        # energy consumed from timestamp to timestamp
        self.bills_consumed = self.data.diff()

        # Remove nans from differencing procedure
        self.bills_consumed = self.bills_consumed.dropna()

    def process_data(self):
        """
        Processes the raw energy bills and cost datasets
        (1) Convert timestamp data into consistent format
        (2) Check for Nans or missing data to predict (linear interpolation or forecasting)
        (3) Compute bills_consumed attribute - energy consumed from month to month
        (4) Remove Nans from differencing and resent index values of bills_consumed
        (5) Extract number of days information from timestamp data as integer
        (6) Determine the running total cost
        (7) Create processed_data attribute - combine bills_consumed and cost and reformat
        (8) Estimate approximate daily usage of energy bills and associated cost
        (9) Estimate approximate average daily usage of energy bills and associated cost for each month
        :return:
        """

        # (1) Try and convert the date data to a proper datetime format for the dataset
        self.data[self.data.columns.values[0]] = pd.to_datetime(
            self.data[self.data.columns.values[0]], format="%d/%m/%Y")
        self.cost[self.cost.columns.values[0]] = pd.to_datetime(
            self.cost[self.cost.columns.values[0]], format="%d/%m/%Y")

        # (2) Checking to see if any Nans or missing values are present in the raw dataset
        if self.data['Electricity'].isna().any():
            nan_id = self.data.loc[self.data['Electricity'].isna() == True]
            # print(nan_id)

            # current fudge fix - linear interpolation
            self.data['Electricity'] = self.data['Electricity'].interpolate()
            self.data['Gas'] = self.data['Gas'].interpolate()
            # print(self.data['Electricity'].interpolate())
            # print(self.data['Gas'].interpolate())

            # alternate approach is to load in the model and forecast the data
            # forecast/predict the data missing for the associated timestamp <<< up to here
            # model_filename = 'model.sav'
            # model = pickle.load(model, open(model_filename, 'wb'))
            # forecast_time = pd.to_datetime("28/06/2023", format="%d/%m/%Y")
            # print(model.predict(nan_id['Time'].reshape(-1,1)))

        # (3) Compute energy consumed from timestamp to timestamp (difference)
        self.bills_consumed = self.data.diff()

        # (4) Remove nans from differencing procedure
        self.bills_consumed = self.bills_consumed.dropna().reset_index()
        self.bills_consumed.pop('index')

        # (5) Extract integer day information from timedelta
        # meter_reading.bills_consumed = meter_reading.bills_consumed.rename(columns={"Time": "Days"})
        self.bills_consumed["Days"] = self.bills_consumed["Time"].dt.days

        # (6) Determine the running total cost
        self.cost['Running Total'] = self.cost['Cost'].cumsum()

        # (7) Combine bills consumed and cost datasets & remove unnecessary data columns
        self.processed_data = pd.concat([self.bills_consumed, self.cost],
                                        axis=1)  # .T.drop_duplicates().T #ignore_index=True)
        self.processed_data = self.processed_data[["Date", "Days", "Gas", "Electricity", "Cost", "Running Total"]]

        # (8) Create subset data of processed_data for modelling purposes (multi-linear regression)
        self.data2model = self.processed_data[["Gas", "Electricity", "Running Total"]].copy()
        self.data2model["Gas"] = self.data2model["Gas"].cumsum()
        self.data2model["Electricity"] = self.data2model["Gas"].cumsum()

        # (9) Estimate approximate average daily usage of energy bills and associated cost for each month
        self.daily_usage = self.processed_data[["Date"]].copy()
        self.daily_usage["Gas"] = self.processed_data["Gas"] / self.processed_data["Days"]
        self.daily_usage["Electricity"] = self.processed_data["Electricity"] / self.processed_data["Days"]
        self.daily_usage["Cost"] = self.processed_data["Cost"] / self.processed_data["Days"]

        # (10) Reassign the timestamp vector from the cost dataset to bills_consumed
        # This aligns the monthly usages with the corresponding cost
        self.bills_consumed['Time'] = self.cost['Date']

        # Write processing step to log file
        self.log_file(['\nRaw energy bills and costs data has been processed\n'])

    def visualise_data(self):
        """
        Visualise the amount of gas and electricity used
        (1) time histories of raw absolute energy & cost data
        (2) time histories of bills consumed
        (3) State-space of electricity, gas and cost dataset
        (3) Distributions (box plot & histograms) of bills consumed
        :return:
        """

        # time-series plot raw usage from month to month
        plt.figure(1)
        plt.subplot(3, 1, 1)
        plt.scatter(self.data['Time'], self.data['Gas'])
        plt.ylabel("Gas [m\u00b3]")

        plt.subplot(3, 1, 2)
        plt.scatter(self.data["Time"], self.data["Electricity"])
        plt.ylabel("Electricity [kWh]")

        plt.subplot(3, 1, 3)
        plt.plot(self.cost['Date'], self.cost['Running Total'], linestyle='None', marker='o', markersize=5)
        plt.xlabel("Time [dd/mm/yyyy]")
        plt.ylabel("Cost [£]")

        # time-series plot of bills usage from month to month
        plt.figure(2)
        plt.subplot(3, 1, 1)
        plt.plot(self.bills_consumed['Time'], self.bills_consumed['Electricity'], linestyle='None', marker='o',
                 markersize=5)
        plt.ylabel("Electricity [kWh]")

        plt.subplot(3, 1, 2)
        plt.plot(self.bills_consumed['Time'], self.bills_consumed['Gas'], linestyle='None', marker='o', markersize=5)
        plt.ylabel("Gas [m\u00b3]")

        plt.subplot(3, 1, 3)
        plt.plot(self.cost['Date'], self.cost['Cost'], linestyle='None', marker='o', markersize=5)
        plt.xlabel("Time [dd/mm/yyyy]")
        plt.ylabel("Cost [£]")

        plt.figure(3)
        plt.plot(self.cost['Date'], self.cost['Cost'], linestyle='None', marker='o', markersize=5, color='k',
                 label='monthly')
        plt.plot(self.cost['Date'], self.cost['Running Total'], linestyle='None', marker='o', markersize=5, color='b',
                 label='running total')
        plt.xlabel("Time")
        plt.ylabel("Cost [£]")
        plt.legend()

        # visualise the state space - electricity & gas (bills consumed)
        plt.figure(4)
        plt.plot(self.bills_consumed['Electricity'], self.bills_consumed['Gas'], linestyle='None', marker='o',
                 markersize=5)
        plt.xlabel("Electricity [kWh]")
        plt.ylabel("Gas [m\u00b3]")

        # visualise full state space - electricity & gas (bills consumed) and cost
        fig = plt.figure(5)
        ax = fig.add_subplot(projection='3d')
        ax.plot(self.bills_consumed["Gas"], self.bills_consumed["Electricity"], self.cost['Cost'],
                'k', linestyle='None', marker='o', markersize=5)
        ax.set_xlabel("Gas [m\u00b3]")
        ax.set_ylabel("Electricity [kWh]")
        ax.set_zlabel("Cost [£]")

        # number of bins for histograms
        u_bins = 4

        plt.figure(6)
        plt.subplot(1, 2, 1)
        plt.boxplot(self.bills_consumed['Gas'])
        plt.ylabel("Gas [m\u00b3]")
        plt.subplot(1, 2, 2)
        plt.hist(self.bills_consumed['Gas'], bins=u_bins)
        plt.xlabel("Gas [m\u00b3]")

        plt.figure(7)
        plt.subplot(1, 2, 1)
        plt.boxplot(self.bills_consumed['Electricity'])
        plt.ylabel("Electricity [kWh]")
        plt.subplot(1, 2, 2)
        plt.hist(self.bills_consumed['Electricity'], bins=u_bins)
        plt.xlabel("Electricity [kWh]")

        plt.figure(8)
        plt.subplot(1, 2, 1)
        plt.boxplot(self.cost['Cost'])
        plt.ylabel("Cost [£]")
        plt.subplot(1, 2, 2)
        plt.hist(self.cost['Cost'], bins=u_bins)
        plt.xlabel("Cost [£]")
        plt.show()

    def visualise_raw_data(self):
        """
        Visualises the raw data as a compact time history: Gas, electricity and cost
        :return:
        """

        # time-series plot raw usage from month to month
        plt.figure(1)
        plt.subplot(3, 1, 1)
        plt.scatter(self.data['Time'], self.data['Gas'])
        plt.ylabel("Gas [m\u00b3]")

        plt.subplot(3, 1, 2)
        plt.scatter(self.data["Time"], self.data["Electricity"])
        plt.ylabel("Electricity [kWh]")

        plt.subplot(3, 1, 3)
        plt.plot(self.cost['Date'], self.cost['Running Total'], linestyle='None', marker='o', markersize=5)
        plt.xlabel("Time [dd/mm/yyyy]")
        plt.ylabel("Cost [£]")
        plt.show()

    def visualise_processed_data(self):
        """
        Visualises the processed gas, electricity and cost data
        - Compact time-history
        - 2D state-space: Gas & electricity
        - 3D state-space: Gas, electricity & cost
        :return:
        """

        # time-series plot of bills usage from month to month
        plt.figure(1)
        plt.subplot(3, 1, 1)
        plt.plot(self.bills_consumed['Time'], self.bills_consumed['Electricity'], linestyle='None', marker='o',
                 markersize=5)
        plt.ylabel("Electricity [kWh]")

        plt.subplot(3, 1, 2)
        plt.plot(self.bills_consumed['Time'], self.bills_consumed['Gas'], linestyle='None', marker='o', markersize=5)
        plt.ylabel("Gas [m\u00b3]")

        plt.subplot(3, 1, 3)
        plt.plot(self.cost['Date'], self.cost['Cost'], linestyle='None', marker='o', markersize=5)
        plt.xlabel("Time [dd/mm/yyyy]")
        plt.ylabel("Cost [£]")

        plt.figure(2)
        plt.plot(self.cost['Date'], self.cost['Cost'], linestyle='None', marker='o', markersize=5, color='k',
                 label='monthly')
        plt.plot(self.cost['Date'], self.cost['Running Total'], linestyle='None', marker='o', markersize=5, color='b',
                 label='running total')
        plt.xlabel("Time")
        plt.ylabel("Cost [£]")
        plt.legend()

        # visualise the state space - electricity & gas (bills consumed)
        plt.figure(3)
        plt.plot(self.bills_consumed['Electricity'], self.bills_consumed['Gas'], linestyle='None', marker='o',
                 markersize=5)
        plt.xlabel("Electricity [kWh]")
        plt.ylabel("Gas [m\u00b3]")

        # visualise full state space - electricity & gas (bills consumed) and cost
        fig = plt.figure(4)
        ax = fig.add_subplot(projection='3d')
        ax.plot(self.bills_consumed["Gas"], self.bills_consumed["Electricity"], self.cost['Cost'],
                'k', linestyle='None', marker='o', markersize=5)
        ax.set_xlabel("Gas [m\u00b3]")
        ax.set_ylabel("Electricity [kWh]")
        ax.set_zlabel("Cost [£]")
        plt.show()

    def visualise_distributions(self):
        """
        Visualises the data distributions of the gas, electricity and cost data
        - As a box and whisker plot
        - As a histogram
        :return:
        """
        # number of bins for histograms
        u_bins = 4

        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.boxplot(self.bills_consumed['Gas'])
        plt.ylabel("Gas [m\u00b3]")
        plt.subplot(1, 2, 2)
        plt.hist(self.bills_consumed['Gas'], bins=u_bins)
        plt.xlabel("Gas [m\u00b3]")

        plt.figure(2)
        plt.subplot(1, 2, 1)
        plt.boxplot(self.bills_consumed['Electricity'])
        plt.ylabel("Electricity [kWh]")
        plt.subplot(1, 2, 2)
        plt.hist(self.bills_consumed['Electricity'], bins=u_bins)
        plt.xlabel("Electricity [kWh]")

        plt.figure(3)
        plt.subplot(1, 2, 1)
        plt.boxplot(self.cost['Cost'])
        plt.ylabel("Cost [£]")
        plt.subplot(1, 2, 2)
        plt.hist(self.cost['Cost'], bins=u_bins)
        plt.xlabel("Cost [£]")
        plt.show()

    def compute_pdf(self):
        """
        Evaluates the probability density function of the data consumed from month to month as 1d and 2d
        representations.
        NEEDS RETHINKING
        :return:
        """

        # 1D PDFs
        # Non-parametric: Kernel density estimation - Gaussian model distribution
        model = KernelDensity(bandwidth=2, kernel='gaussian')
        sample = self.bills_consumed["Time"].to_numpy().reshape((len(self.bills_consumed), 1))
        model.fit(sample)
        gas_values = np.asarray(
            [value for value in range(int(self.bills_consumed["Gas"].min()), int(self.bills_consumed["Gas"].max()))])
        self.pdf.gas.values = gas_values.reshape((len(gas_values), 1))
        gas_probabilities = model.score_samples(gas_values)
        self.pdf.gas.probabilities = np.exp(gas_probabilities)
        elec_values = np.asarray(
            [value for value in
             range(int(self.bills_consumed["Electricity"].min()), int(self.bills_consumed["Electricity"].max()))])
        self.pdf.elec.values = elec_values.reshape((len(elec_values), 1))
        elec_probabilities = model.score_samples(elec_values)
        self.pdf.elec.probabilities = np.exp(elec_probabilities)

        # 2D PDF
        # Extract x and y
        x = self.bills_consumed["Gas"].to_numpy()
        y = self.bills_consumed["Electricity"].to_numpy()

        # Define the borders
        deltaX = (max(x) - min(x)) / 10
        deltaY = (max(y) - min(y)) / 10
        self.pdf.TwoD.xmin = min(x) - deltaX
        self.pdf.TwoD.xmax = max(x) + deltaX
        self.pdf.TwoD.ymin = min(y) - deltaY
        self.pdf.TwoD.ymax = max(y) + deltaY
        print(self.pdf.TwoD.xmin, self.pdf.TwoD.xmax, self.pdf.TwoD.ymin, self.pdf.TwoD.ymax)

        # Create meshgrid
        self.pdf.TwoD.xx, self.pdf.TwoD.yy = np.mgrid[self.pdf.TwoD.xmin:self.pdf.TwoD.xmax:100j,
                                             self.pdf.TwoD.ymin:self.pdf.TwoD.ymax:100j]

        positions = np.vstack([self.pdf.TwoD.xx.ravel(), self.pdf.TwoD.yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        self.pdf.TwoD.f = np.reshape(kernel(positions).T, self.pdf.TwoD.xx.shape)

        # 2D PDF contour plot
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.gca()
        # ax.set_xlim(self.pdf.TwoD.xmin, self.pdf.TwoD.xmax)
        # ax.set_ylim(self.pdf.TwoD.ymin, self.pdf.TwoD.ymax)
        # cfset = ax.contourf(self.pdf.TwoD.xx, self.pdf.TwoD.yy, self.pdf.TwoD.f, cmap='coolwarm')
        # ax.imshow(np.rot90(self.pdf.TwoD.f), cmap='coolwarm',
        #           extent=[self.pdf.TwoD.xmin, self.pdf.TwoD.xmax, self.pdf.TwoD.ymin, self.pdf.TwoD.ymax])
        # cset = ax.contour(self.pdf.TwoD.xx, self.pdf.TwoD.yy, self.pdf.TwoD.f, colors='k')
        # ax.clabel(cset, inline=1, fontsize=10)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # plt.title('2D Gaussian Kernel density estimation')
        # plt.show()

    def print_statistics(self):
        """
        Prints out the general statistic distribution of the bills consumed attribute

        Compute the overall statistics of the monthly amounts used and print out mean +- std
        :return:
        """

        stats_string_list = ['\nEnergy bill usage statistics:',
                             f'\nTotal gas consumption: {round(self.bills_consumed["Gas"].sum(), 2)} m\u00b3',
                             f'\nAverage gas consumption: {round(self.bills_consumed["Gas"].mean(), 2)} '
                             f'\u00B1 {round(self.bills_consumed["Gas"].std(), 2)} m\u00b3',
                             f'\nTotal electricity consumption: {round(self.bills_consumed["Electricity"].sum(), 2)} Kwh'
                             f'\nAverage electricity consumption: {round(self.bills_consumed["Electricity"].mean(), 2)} '
                             f'\u00B1 {round(self.bills_consumed["Electricity"].std(), 2)} kWh',
                             f'\nTotal energy bills spent: £{str(round(self.cost["Cost"].sum(), 2))}',
                             f'\nAverage energy bills spent: £{str(round(self.cost["Cost"].mean(), 2))} '
                             f'\u00B1 £{str(round(self.cost["Cost"].std(), 2))} \n']

        for string in stats_string_list:
            print(string)

        # write processed_data statistics summary to log file
        self.log_file(stats_string_list)

    def number_of_clusters(self):
        """
        Determine the number of clusters (groups) inherent in the data based on KMeans cluster elbow technique.
        The gradient of the sum of squared error (sse) function is minimised to determine the location of significant
        reduction in error.
        :return: clusters_num
        """

        # Perform Kmeans cluster elbow technique
        sse = {}  # use the sum of squared error as a quantitative error metric
        for k in range(1, 5):
            kmeans = KMeans(n_clusters=k, max_iter=1000).fit(
                self.processed_data[['Gas', 'Electricity', 'Cost']])
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

    def compute_cluster_model(self):
        """
        Create Kmeans cluster model using processed energy data: [gas, electricity, cost] is clustered into groups
        based on an elbow technique analysis to determine the nominal number of clusters
        :return: cluster_model
        """

        # determine nominal number of clusters
        clusters_num = self.number_of_clusters()

        # Develop model and determine cluster groups/labels
        model = KMeans(n_clusters=clusters_num)
        model.fit(self.processed_data[['Gas', 'Electricity', 'Cost']])
        labels = model.predict(self.processed_data[['Gas', 'Electricity', 'Cost']])

        self.cluster_model = model
        self.cluster_labels = labels

        return labels

    def model_data(self, x, y, deg, log_text):
        """
        Model raw input data using simple regression for SKLearn
        Caters for both simple and multi-linear regression models

        Updated to output model fit performance to log file
        :return:
        """
        # TODO
        # Model using some form of ARIMA/SARIMA model - need more data for that though

        # transform data to numpy arrays
        X = x.to_numpy()
        Y = y.to_numpy()

        # determine dimensionality of input feature vector - determining simple linear or multi-linear regression
        if len(x.shape) > 1:
            X = x.iloc[:, 0:-1]
        elif len(x.shape) == 1:
            X = x.to_numpy().reshape((-1, 1))

        # create model
        transformer = PolynomialFeatures(degree=deg, include_bias=False)
        X_train = transformer.fit_transform(X)
        model = LinearRegression().fit(X_train, Y)
        y_pred = model.predict(X_train)

        print('Model: r-squared value: ', r2_score(y, y_pred))

        model_string_list = [f'Energy bills modelling {log_text} performance:\n',
                             f'{len(x.shape)}-variable model, polynomial: order {deg}'
                             f'\nr-squared value: {round(r2_score(y, y_pred), 3)}'
                             ]

        # write model fit summary to log file
        self.log_file(model_string_list)

        # TODO: Add functionality to predict electricity and gas for a specific timestamp
        # y_pred = model.predict(x)
        return model, y_pred

    def reduce_dim(self, X):
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
        plt.title('Scree plot: [Days,Gas,Electricity,Cost]')
        plt.savefig('Figures\\Energy_bills_pca_dimensionality_analysis.png', dpi=800)
        plt.show()

        # write PCA dimensional modelling analysis to log file
        model_string_list = ['\nEnergy bills dimensionality analysis:\n',
                             pca_list[0] + ': ' + str(round(pca.explained_variance_ratio_[0] * 100, 3)) + '%',
                             '\n' + pca_list[1] + ': ' + str(round(pca.explained_variance_ratio_[1] * 100, 3)) + '%',
                             '\n' + pca_list[2] + ': ' + str(round(pca.explained_variance_ratio_[2] * 100, 3)) + '%']
        self.log_file(model_string_list)

        # rescale the dataset first
        # pca = PCA()
        # pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
        # Xt = pipe.fit_transform(X)

        return pca, Xt

    def log_file(self, string_to_add):
        """
        Create a log file of all the data processing, modelling and statistics of energy bills consumed
        Call log file within other functions to pass in processed data, model fit and overall metadata
        :return:
        """

        if os.path.exists(self.log_filename):
            # ensure new line is started in log file every time function is called to append correctly
            new_line = ['\n']
            with open(self.log_filename, 'a') as f:
                f.writelines(new_line + string_to_add)

        else:
            with open(self.log_filename, 'w') as f:
                f.writelines(string_to_add)

    def load_weather_data(self):
        """
        Carry out the WeatherData class method
        (1) Load in weather data
        (2) Parse weather data
        (3) Visualise raw weather datasets
        (4) Visualise weather data columns distributions
        (5) Compute correlations of weather data columns
        (6) Compute mean cycles of each weather data column
        (7) Perform PCA - data dimensionality analysis
        :return:
        """

        # self.weather_filename
        # (1) Instantiate WeatherData class
        filename = 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/stationdata/suttonboningtondata.txt'
        self.weather = WeatherData(filename)

        # (2) Parse weather data into usable data format
        plotopt = 0
        self.weather.parse_weather_data(plotopt)

        # # (3) Visualise raw weather datasets
        self.weather.visualise_raw_weather_data()
        #
        # # (4) Visualise weather data columns distributions
        # self.weather.visualise_weather_data_distributions()
        #
        # # (5) Compute correlations of weather data columns
        self.weather.compute_correlations()

        # (6) Compute mean cycles of each weather data column
        self.weather.compute_mean_cycle()

        # (7) Perform PCA - dimensionality analysis data
        X = self.weather.Wdata[['tmin', 'tavg', 'tmax', 'rain', 'sun']].copy().to_numpy()
        # pre-process data to remove NaNs and estimate their values based on multivariate imputation - estimator model
        # Note that IterativeImputer is experimental and syntax/format might change
        imp = IterativeImputer(max_iter=10, random_state=0)
        imp.fit(X)
        pca, Xt = self.weather.reduce_dim(imp.transform(X))

        weather_string = ['Meteorological data loaded in and processed. '
                          f'\n{filename}']
        self.log_file(weather_string)

    def filter_weather_data(self):
        """
        Filter weather data (monthly temperature, rain and sun data) wand align with energy bills timescale/frame
        :return:
        """

        # idx = self.processed_data['Date'].min() <= self.weather.Wdata['date'] <= self.processed_data['Date'].max()
        # (self.weather.Wdata['date'] >= self.processed_data['Date'].min()) and (self.weather.Wdata['date'] <=
        # self.processed_data['Date'].max())
        idx = self.weather.Wdata['date'] >= self.processed_data['Date'].min()
        weather_datafilt = self.weather.iloc[idx, :]

        if len(weather_datafilt) == len(self.processed_data['Date']):
            print("Weather data successfully filtered to energy bills timeframe")
        else:
            print("Weather data unsuccessfully filtered to energy bills timeframe")

        return idx

    def analyse_energy_weather_datasets(self):
        """
        Evaluate correlations between:
        - Electricity usage - average temperature
        - Electricity usage - rain
        - Gas usage - average temperature
        - Gas usage - rain
        :return:
        """

        # correlate weather temperature & rain data with gas & electricity usage
        # create a figure of electricity against temperature
        # create a figure of gas against temperature
        idx = filter_weather_data(self)

        # Electricity usage analysis
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(self.processed_data['Date'], self.processed_data['Electricity'], 'k', linestyle='None',
                 marker='o', markersize=5)
        plt.ylabel('Electricity [kWhr]')
        plt.subplot(3, 1, 2)
        plt.plot(self.weather.Wdata['date'][idx], self.weather.Wdata['tavg'][idx], 'b', linestyle='None',
                 marker='o', markersize=5)
        plt.ylabel("Temp. [$^\circ$C]")
        plt.subplot(3, 1, 3)
        plt.plot(self.weather.Wdata['date'], self.weather.Wdata['rain'][idx], 'b', linestyle='None',
                 marker='o', markersize=5)
        plt.ylabel("Rain [mm]")

        print('\nElectricity usage - temperature: ' +
              str(round(self.processed_data['Electricity'].corr(self.weather.Wdata['tavg'][idx]))))
        print('\nElectricity usage - rain: ' +
              str(round(self.processed_data['Electricity'].corr(self.weather.Wdata['rain'][idx]))))

        # Gas usage analysis
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(self.processed_data['Date'], self.processed_data['Gas'], 'k', linestyle='None',
                 marker='o', markersize=5)
        plt.ylabel('Gas [m\u00b3]')
        plt.subplot(3, 1, 2)
        plt.plot(self.weather.Wdata['date'][idx], self.weather.Wdata['tavg'][idx], 'b', linestyle='None',
                 marker='o', markersize=5)
        plt.ylabel("Temp. [$^\circ$C]")
        plt.subplot(3, 1, 3)
        plt.plot(self.weather.Wdata['date'], self.weather.Wdata['rain'][idx], 'b', linestyle='None',
                 marker='o', markersize=5)
        plt.ylabel("Rain [mm]")

        print('\nGas usage - temperature: ' +
              str(round(self.processed_data['Gas'].corr(self.weather.Wdata['tavg'][idx]))))
        print('\nGas usage - rain: ' +
              str(round(self.processed_data['Gas'].corr(self.weather.Wdata['rain'][idx]))))

    def main(self):

        # (1) load in most recent dataset (.txt) file as a table/spreadsheet
        self.data = self.load_file(self.filename)
        self.cost = self.load_file('energy_bills.txt')

        # (2) Prompt program user to input gas and electricity meter readings in (Gas in m^3 & electricity in kwh)
        self.input_new_data()

        # (3) Write new data to existing .txt file
        self.write_file()

        # (4)
        # self.compute_pdf()

        # (5) visualise the amount of energy (Electricity & Gas used)
        self.visualise_data()

        # (6) print out statistical information
        self.print_statistics()

        # Press the green button in the gutter to run the script.
        # if __name__ == 'MeterReadings':
        #     MeterReadings.main()
