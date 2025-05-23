"""
@file   helpers.py
@brief  Helper functions for the stats visualization scripts like fitting curves and setting log ticks on plots.
@author Jan Zdeněk (xzdene01)
@date   29/3/2025

@project Aproximace násobiček pomocí CGP
@course  BIN - Biologií inspirované počítače
@faculty Faculty of Information Technology, Brno University of Technology
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def neg_exp_offset(x, a, b, c):
    return c + a * np.exp(-b * x)


def get_trend_exp(x_series: pd.Series, y_series: pd.Series) -> callable:
    x_data = x_series.to_numpy()
    y_data = y_series.to_numpy()

    # Initial guess for the parameters
    a_guess = y_data[0] - y_data[-1]
    b_guess = 1.0
    c_guess = y_data[-1]
    initial_guess = [a_guess, b_guess, c_guess]

    # Fit the function to the data
    popt, _ = curve_fit(neg_exp_offset, x_data, y_data, p0=initial_guess)
    a_fit, b_fit, c_fit = popt
    
    return lambda x: neg_exp_offset(x, a_fit, b_fit, c_fit), (a_fit, b_fit, c_fit)

def get_trend_poly(x_series: pd.Series, y_series: pd.Series, degree: int = 2) -> callable:
    x_series = x_series.to_numpy().reshape(-1, 1)
    y_series = y_series.to_numpy()

    polynomial_features = PolynomialFeatures(degree)
    linear_regression = LinearRegression()

    pipeline = Pipeline([
        ("polynomial_features", polynomial_features),
        ("linear_regression", linear_regression)
    ])
    pipeline.fit(x_series, y_series)

    coef = pipeline.named_steps["linear_regression"].coef_
    intercept = pipeline.named_steps["linear_regression"].intercept_
    return lambda x: pipeline.predict(x.reshape(-1, 1)), (coef, intercept)


def set_log_ticks(ax: plt.Axes, num_ticks: int, axis: str = "x", decimals: int = 3):
    if axis == "x":
        min, max = ax.get_xlim()
    else:
        min, max = ax.get_ylim()

    log_ticks = np.linspace(np.log10(min), np.log10(max), num_ticks)
    ticks = 10 ** log_ticks

    if axis == "x":
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter(f"%.{decimals}f"))
    else:
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.set_yticks(ticks)
        ax.get_yaxis().set_major_formatter(plt.FormatStrFormatter(f"%.{decimals}f"))