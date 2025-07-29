import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import numpy as np


# create plotter function
def plot():
    ax.clear()
    x = np.random.randint(0,10,10)
    y = np.random.randint(0, 10, 10)
    ax.scatter(x,y)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    canvas.draw()

# Initialise Tkinter
root = tk.Tk()
fig = Figure()
ax = fig.add_subplot()

# Tkinter Application
frame = tk.Frame(root)
label = tk.Label(text = "Matplotlib + Tkinter!")
label.config(font=("Courier", 32))
label.pack()

canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().pack()

toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
toolbar.update()
toolbar.pack()
toolbar.pack(anchor="w", fill= tk.X)

frame.pack()

tk.Button(frame, text="Plot Graph", command=plot).pack(pady=10)
root.mainloop()