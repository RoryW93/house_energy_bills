"""
This script analyses energy consumption used by Dr Rory White and Ms Catherine Tuckey
at 65 Salisbury Street, Beeston, NG9 2EQ

House details:
3 bedrooms
2 bathrooms

Date: 01/05/2023
Author: Dr Rory White
Location: Nottingham, UK

added smart meter readings text file - 02/07/23
updated 29/07/23
updated 02/08/23
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from MeterReadings import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import pickle
from datetime import date

# (0) Determine input data set to analyse
file_input = input('smart or monthly data: ')
energy_cost_input = input('Is the energy usage cost data available? ')

if file_input.lower() == 'smart':
    filename = 'Data\\smart_meter_readings.txt'
elif file_input.lower() == 'monthly':
    filename = 'Data\\meter_readings_monthly.txt'
else:
    print('Incorrect input format. Loading Monthly meter readings data...')
    filename = 'Data\\meter_readings_monthly.txt'

if energy_cost_input.lower() == 'y':
    energy_bills = 'Data\\energy_bills.txt'

# (1) Instantiate objects & load in datasets
mode = 2
meter_reading = MeterReadings(filename, mode)
meter_reading.data = meter_reading.load_file(filename)
if energy_cost_input.lower() == 'y':
    meter_reading.cost = meter_reading.load_file(energy_bills)

# add functionality to calculate energy bill based on usage
# Electricity calculated as: Usage (kWh) * cost/kWh + cost/day
# winter 2023 - 30.857p/kWh + 47.912p/day
# summer 2023 - 28.078p/kWh + 47.912p/day
# (Octopus -  24.55p/kWh + 46.34p/day)

# Gas calculated as: Usage (m^3) * Volume Correction * Calorific Value (J/m^3) / 3.6 (kWh/J) * cost/kWh + cost/day
# - average Volume Correction - 1.02264
# - average Calorific Value - 39.5J/m^3
# winter 2023-2024 - 5.72p/kWh + 26.16p/day

# (2) Load weather data for analysis
meter_reading.load_weather_data()

# (3) Process energy bills data
meter_reading.process_data()
print("Processed data summary:")
print(meter_reading.processed_data)
print("Daily usage summary:")
print(meter_reading.daily_usage)

# (4) Visualise the data
## meter_reading.visualise_data()
meter_reading.visualise_raw_data()
meter_reading.visualise_processed_data()
# meter_reading.visualise_distributions()

# (5) Create cluster model of processed energy bills data - determine number of cluster groups
labels = meter_reading.compute_cluster_model()
num_of_labels = np.unique(labels)

plt.figure(1, figsize=(15, 20))
plt.subplot(1,3,1)
plt.scatter(meter_reading.bills_consumed["Gas"], meter_reading.bills_consumed["Electricity"], c=labels)
plt.xlabel("Gas [m\u00b3]")
plt.ylabel("Electricity [kWh]")
plt.subplot(1,3,2)
plt.scatter(meter_reading.bills_consumed["Gas"], meter_reading.cost['Cost'], c=labels)
plt.xlabel("Gas [m\u00b3]")
plt.ylabel("Cost [£]")
plt.subplot(1,3,3)
plt.scatter(meter_reading.bills_consumed["Electricity"], meter_reading.cost['Cost'], c=labels)
plt.xlabel("Electricity [kWh]")
plt.ylabel("Cost [£]")
plt.savefig('Figures\\energy_bills_cluster.png',dpi=800)
plt.show()

# predict model labels based on synthetic data
# new_labels = model.predict(new_samples)

# (6) Compute statistics
meter_reading.print_statistics()

# (7) Model data - individually and combined datasets
deg = 1

# (7a) Model electrical data
model_e, y_pred_e = meter_reading.model_data(meter_reading.data['Time'],
                                             meter_reading.data['Electricity'], deg, 'Electricity_TH')

# (7b) Model gas data
model_g, y_pred_g = meter_reading.model_data(meter_reading.data['Time'], meter_reading.data['Gas'], 5, 'Gas_TH')

# (7c) Model gas, electrical and cost data combined: Cost = b0 + b1*electrical +b2*gas
# For multiple linear regression model, require the cumulative amounts of gas, electricity and cost consumed
model_GEC, y_pred_cost = meter_reading.model_data(meter_reading.data2model[["Gas","Electricity"]],
                                                  meter_reading.data2model["Running Total"], 2, 'Gas_Electricity_Cost')

# (7d) check individually modelled data
plt.figure(1, figsize=(7, 5))
# Electricity linear model
plt.subplot(2, 1, 1)
plt.plot(meter_reading.data['Time'].to_numpy().reshape(-1,1), meter_reading.data['Electricity'].to_numpy(), 'k', label='elec_data', linestyle='None', marker='o', markersize=5)
plt.plot(meter_reading.data['Time'].to_numpy().reshape(-1,1), y_pred_e, 'r--', label='fit')  #: a=%5.3f, b=%5.3f' % tuple(model_e.coef_))
plt.ylabel('Electricity consumption [kWh]')
plt.legend()

# Gas linear model
plt.subplot(2, 1, 2)
plt.plot(meter_reading.data['Time'].to_numpy().reshape(-1,1), meter_reading.data['Gas'].to_numpy(), 'k', label='gas_data', linestyle='None', marker='o', markersize=5)
plt.plot(meter_reading.data['Time'].to_numpy().reshape(-1,1), y_pred_g, 'r--', label='fit')  #: a=%5.3f, b=%5.3f' % tuple(model_e.coef_))
plt.xlabel('Time')
plt.ylabel('Gas consumption [m\u00b3]')
plt.legend()
plt.savefig('Figures\\separate_electricity_gas_models.png',dpi=800)
plt.show()

# (7e) Visualise gas and electricity as a function of running total cost
fig = plt.figure(2)
ax = fig.add_subplot(projection='3d')
ax.plot(meter_reading.data2model["Gas"], meter_reading.data2model["Electricity"], meter_reading.data2model["Running Total"], 'k', label='data',  linestyle='None', marker='o', markersize=5)
ax.plot(meter_reading.data2model["Gas"], meter_reading.data2model["Electricity"], y_pred_cost, 'r--', label='fit')
ax.set_xlabel("Gas [m\u00b3]")
ax.set_ylabel("Electricity [kWh]")
ax.set_zlabel("Cost [£]")
plt.savefig('Figures\\combined_electricity_gas_model.png',dpi=800)
plt.show()

# (7f) Forecast data - needs updating -> somehow reshape the data
# .astype('datetime64[D]').values
# forecast_time = pd.to_datetime("28/06/2023", format="%d/%m/%Y").to_numpy()
# print(model_e.predict(forecast_time.reshape(1,-1)))
# aa = 1

# (7g) Save models
model_filename1 = 'Models\\electricity_model.pkl'
model_filename2 = 'Models\\gas_model.pkl'
model_filename3 = 'Models\\gas_electric_model.pkl'
pickle.dump(model_e, open(model_filename1, 'wb'))
pickle.dump(model_g, open(model_filename2, 'wb'))
pickle.dump(model_GEC, open(model_filename3, 'wb'))

# pickled_model = pickle.load(open('Models\\model.pkl', 'rb'))

# (8) Reduce dimensionality of data
# determine dimensionality of the dataset - for the gas, electricity & cost
X = meter_reading.processed_data[["Gas","Electricity","Cost"]].copy().to_numpy()
# X = meter_reading.processed_data[["Days","Gas","Electricity","Cost"]].copy().to_numpy()
pipe, Xt = meter_reading.reduce_dim(X)

# visualise the reduced dataset
plt.figure(3)
plt.plot(meter_reading.processed_data["Date"],Xt[:,0], 'k',  linestyle='None', marker='o', markersize=5)
plt.xlabel("Time [yyyy/mm]")
plt.ylabel("Reduced dataset - Single variable")
plt.savefig('Figures\\reduced_SDOF_energybills_data.png', dpi=800)
plt.show()

# (8b) Model the reduced dataset as a function of time -

# (9) Filter and align meteorological and energy bills data
# idx = meter_reading.filter_weather_data()

aa = 1