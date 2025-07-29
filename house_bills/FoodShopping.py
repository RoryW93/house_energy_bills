"""
Take in weekly shopping cost data as a total
Compare against a weekly goal and evaluate difference - positive (underspent) or negative (overspent)
Identify month to month spending - positive, negative or zero
Evaluate any trends

Use some of the functionality from MeterReadings
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.neighbors import KernelDensity
from MeterReadings import MeterReadings


class FoodShopping(MeterReadings):
    pass

    # Constructor
    def __init__(self, filename):  # or *argv for a variable number of arguments

        # check if filename is passed through or as a string variable
        if not filename or filename is None:
            self.filename = 'food_shopping.xlsx'
        elif isinstance(filename, str):
            self.filename = filename

        self.data = None
        self.bills_consumed = None
        self.pdf = None

        super().__init__()

        # perform main method thread
        self.main()

    # take the current year and use to evaluate the correct sheet to perform analysis
