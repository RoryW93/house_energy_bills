"""
Class object to develop GUI application functionality
Needs filling in properly.

Date:
Author: Dr Rory White
Location: Nottingham, UK

To-do:
Integrate MeterReadings class - energy bills post-processing & analysis
Integrate WeatherData class - meteorological (weather) data processing & analysis
"""

from MeterReadings import *  # MeterReadings #* #MeterReadings
import tkinter as tk  # *
from tkinter.filedialog import askopenfile
from tkinter.simpledialog import askstring, askinteger
from tkinter.messagebox import showinfo
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import pickle
from datetime import date
from PIL import Image, ImageTk

#Figure out how to make GUI App into a class object.


class GUIApp(tk.Tk):
    pass

# Re-write all code/functions as class methods - from GUIMain script
    def __init__(self, *argv):
        super().__init__()

        # characterise a config property variable - argv
        # Note *argv is the variable argument variable handler -
        # Define all necessary property variables
        self.config = None
        self.meter_reading = None
        self.weather_data = None

        # characterise a config property variable - argv
        # determine all necessary config property variables
        self.config.energy_data_filename = argv[0]
        self.config.energy_cost_filename = argv[1]
        self.config.weather_data_filename = argv[2]
        self.config.today = argv[3]  # today = date.today().strftime("%d/%m/%Y")
        self.config.plotopt = 1

        # test or check files input are in correct format
        # or automatically load in required data files


# inputs for the GUI
filename = 'Data\\meter_readings_monthly.txt'  # energy bills dataset
energy_bills = 'Data\\energy_bills.txt'  # energy bills cost data
mode = 2  # application mode
instanceMR = MeterReadings(filename, mode)  # instantiate MeterReadings

# instantiate modelling format for GUI app
# 1 - Electricity - time-series
# 2 - Gas - time-series
# 3 - Electricity-Gas-Cost
instanceMR.model = 1

# determine today's date
today = date.today().strftime("%d/%m/%Y")

# start raw data plotting at 1
# 1 - Electricity time-series
# 2 - Gas time-series
# 3 - Cost time-series
plotopt = 1

# create main structure of the GUI app
root = Tk()
root.geometry("800x800")
# root.geometry('+%d+%d' % (350, 100))  # place GUI at x=350, y=10


def open_file():
    browse_text.set("Loading...")
    file = askopenfile(parent=root, mode='rb', filetypes=[("txt file", "*.txt")])
    if file:
        #        mode = askinteger('Model', 'Which mode would you like')
        #        showinfo('Okay', 'Mode: {}'.format(str(mode)))

        # datasets to load - energy bills & cost datasets
        instanceMR.data = instanceMR.load_file(file)
        instanceMR.cost = instanceMR.load_file(energy_bills)

        # process data
        instanceMR.process_data()

        # reset the button text back to Browse
        browse_text.set("Load energy bills")


# plot function is created for plotting the relevant graphs in tkinter window
def plot_raw_data():
    global plotopt
    if plotopt == 1:
        ax.clear()
        ax.scatter(instanceMR.data["Time"], instanceMR.data["Electricity"])
        ax.set_xlabel("Time [yyyy/mm]")
        ax.set_ylabel("Electricity [kWh]")
        canvas.draw()
        plotopt += 1
    elif plotopt == 2:
        ax.clear()
        ax.scatter(instanceMR.data["Time"], instanceMR.data["Gas"])
        ax.set_xlabel("Time [yyyy/mm]")
        ax.set_ylabel("Gas [m\u00b3]")
        canvas.draw()
        plotopt += 1

    elif plotopt == 3:
        ax.clear()
        ax.scatter(instanceMR.cost['Date'], instanceMR.cost['Running Total'])
        ax.set_xlabel("Time [yyyy/mm]")
        ax.set_ylabel("Cost [£]")
        ax.set_ylim([0, round(instanceMR.cost['Running Total'].max(),0)+100])
        canvas.draw()
        plotopt = 1


def model_data():
    # add functionality for choice of data to model
    deg = 1

    if instanceMR.model == 1:
        # Model electrical data
        model_e, y_pred_e = instanceMR.model_data(instanceMR.data['Time'], instanceMR.data['Electricity'], deg)

        model_filename1 = 'Models\\electricity_model.pkl'
        pickle.dump(model_e, open(model_filename1 + today, 'wb'))

    elif instanceMR.model == 2:
        # Model gas data
        model_g, y_pred_g = instanceMR.model_data(instanceMR.data['Time'], instanceMR.data['Gas'], deg)

        model_filename2 = 'Models\\gas_model.pkl'
        pickle.dump(model_g, open(model_filename2 + today, 'wb'))

    elif instanceMR.model == 3:
        # Model gas, electrical and cost data combined: Cost = b0 + b1*electrical +b2*gas
        model_GEC, y_pred_cost = instanceMR.model_data(instanceMR.data2model[["Gas", "Electricity"]],
                                                       instanceMR.data2model["Running Total"],
                                                       deg)
        model_filename3 = 'Models\\gas_electric_model.pkl'
        pickle.dump(model_GEC, open(model_filename3 + today, 'wb'))


def validate_model():
    # TODO: Add functionality to validate and plot the corresponding energy data that has been modelled
    # Need to pickle.load the correct model based on the filename

    if instanceMR.model == 1:
        # check electricity modelled data
        ax.clear()
        ax.plot(instanceMR.data['Time'].to_numpy().reshape(-1, 1), instanceMR.data['Electricity'].to_numpy(),
                'k',
                label='elec_data', linestyle='None', marker='o', markersize=5)
        ax.plot(instanceMR.data['Time'].to_numpy().reshape(-1, 1), y_pred_e, 'r--',
                label='fit')  #: a=%5.3f, b=%5.3f' % tuple(model_e.coef_))
        ax.set_ylabel('Electricity consumption [kWh]')
        ax.legend()
        plt.savefig('Figures\\electricity_model_' + today + '.png', dpi=800)

    elif instanceMR.model == 2:
        # Check Gas modelled data
        ax.clear()
        ax.plot(instanceMR.data['Time'].to_numpy().reshape(-1, 1), instanceMR.data['Gas'].to_numpy(), 'k',
                label='gas_data', linestyle='None', marker='o', markersize=5)
        ax.plot(instanceMR.data['Time'].to_numpy().reshape(-1, 1), y_pred_g, 'r--',
                label='fit')  #: a=%5.3f, b=%5.3f' % tuple(model_e.coef_))
        ax.set_xlabel('Date [mm/yyyy]')
        ax.set_ylabel('Gas consumption [m\u00b3]')
        ax.legend()
        plt.savefig('Figures\\gas_model_' + today + '.png', dpi=800)

    elif instanceMR.model == 3:
        # (6e) Visualise gas and electricity as a function of running total cost
        ax.plot(instanceMR.data2model["Gas"], instanceMR.data2model["Electricity"], instanceMR.data2model["Running "
                                                                                                          "Total"],
                'k', label='data',
                linestyle='None', marker='o', markersize=5)
        ax.plot(instanceMR.data2model["Gas"], instanceMR.data2model["Electricity"], instanceMR.data2model["Running "
                                                                                                          "Total"],
                y_pred_cost, 'r--',
                label='fit')
        ax.set_xlabel("Gas [m\u00b3]")
        ax.set_ylabel("Electricity [kWh]")
        ax.set_zlabel("Cost [£]")
        plt.savefig('Figures\\electricity_gas_cost_model_' + today + '.png', dpi=800)


# Functions to add:
# MeterReadings.display_logo('lightbulb image.jpg', 0, 0)

# place()/grid()/pack() to arrange Frames vertically

# right-hand side area - buttons
RHS_frame = Frame(root, width=300, height=600, bg="#20bebe")
RHS_frame.grid(columnspan=2, rowspan=12)

# add image into GUI
img = Image.open('lightbulb image.jpg')
img = img.resize((200, 200))
img = ImageTk.PhotoImage(img)
img_label = Label(RHS_frame, image=img, bg="white")
img_label.image = img
img_label.grid(column=0, row=0, rowspan=4)


# app title space
Title = Label(RHS_frame, text="Energy bill usage: Analytics", font=("Raleway", 12),
              bg="#20bebe", fg="white", height=2, width=50)
Title.grid(column=0, row=4)

# load energy bills data button
browse_text = StringVar()
# command=lambda: open_file() = for browse_btn
browse_btn = Button(RHS_frame, textvariable=browse_text, command=open_file,
                    font=("Raleway", 12), bg="#20bebe", fg="white", height=1, width=20)
browse_text.set("Load energy bills")
browse_btn.grid(column=0, row=5)

# plot energy usage button
plot_usg_txt = StringVar()
# command=lambda: plot_raw_data()
plot_usg_txt.set("Visualise usage")
plot_usg_btn = Button(RHS_frame, textvariable=plot_usg_txt, command=plot_raw_data,
                      font=("Raleway", 12), bg="#20bebe", fg="white", height=1, width=20)
plot_usg_btn.grid(column=0, row=6)

# print_statistics button
print_stat_txt = StringVar()
# command=lambda: plot_raw_data()
print_stat_txt.set("Compute average usage")
print_stat_btn = Button(RHS_frame, textvariable=print_stat_txt, command=instanceMR.print_statistics,
                        font=("Raleway", 12), bg="#20bebe", fg="white", height=1, width=20)
print_stat_btn.grid(column=0, row=7)

# plot distributions button
plot_dist_txt = StringVar()
# command=lambda: plot_raw_data()
plot_dist_txt.set("Visualise energy distributions")
plot_dist_btn = Button(RHS_frame, textvariable=plot_dist_txt,
                       font=("Raleway", 12), bg="#20bebe", fg="white", height=1, width=20)
plot_dist_btn.grid(column=0, row=8)

# model data button
model_data_txt = StringVar()
# command=lambda: plot_raw_data()
model_data_txt.set("Model data")
model_data_btn = Button(RHS_frame, textvariable=model_data_txt, command=model_data,
                        font=("Raleway", 12), bg="#20bebe", fg="white", height=1, width=20)
model_data_btn.grid(column=0, row=9)

# validate model button
val_data_txt = StringVar()
# command=lambda: plot_raw_data()
val_data_txt.set("Validate data model")
val_data_btn = Button(RHS_frame, textvariable=val_data_txt, command=validate_model,
                      font=("Raleway", 12), bg="#20bebe", fg="white", height=1, width=20)
val_data_btn.grid(column=0, row=10)

# Dimensionality reduction model button
dim_rud_txt = StringVar()
# command=lambda: plot_raw_data()
dim_rud_txt.set("Dim. reduction")
dim_rud_btn = Button(RHS_frame, textvariable=dim_rud_txt, command=instanceMR.reduce_dim,
                      font=("Raleway", 12), bg="#20bebe", fg="white", height=1, width=20)
dim_rud_btn.grid(column=0, row=11)

# load weather data button
load_weath_text = StringVar()
# command=lambda: open_file() = for browse_btn
load_weath_btn = Button(RHS_frame, textvariable=load_weath_text, command=instanceMR.load_weather_data,
                        font=("Raleway", 12), bg="#20bebe", fg="white", height=1, width=20)
load_weath_text.set("Load meteorological data")
load_weath_btn.grid(column=1, row=5)

# visualise weather data button
vis_weath_text = StringVar()
# command=lambda: open_file() = for browse_btn
vis_weath_btn = Button(RHS_frame, textvariable=vis_weath_text,
                        font=("Raleway", 12), bg="#20bebe", fg="white", height=1, width=20)
vis_weath_text.set("Visualise meteorological data")
vis_weath_btn.grid(column=1, row=6)

# create figure
fig = Figure(figsize=(5, 5), dpi=100)
if instanceMR.model != 3:
    ax = fig.add_subplot()
else:
    ax = fig.add_subplot(projection='3d')

# add figure to GUI
canvas = FigureCanvasTkAgg(fig, master=RHS_frame)
# canvas.get_tk_widget().grid(column=2, row=1, sticky=E, padx=100)
canvas.get_tk_widget().grid(column=2, row=0, rowspan=12)

root.mainloop()