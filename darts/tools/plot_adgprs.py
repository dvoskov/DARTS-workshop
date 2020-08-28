import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plot_darts import *

def plot_bhp_adgprs(well_name, adgprs_df, style = '-.', color = "#C41E3A", ax = None):
    
    my_df = pd.DataFrame()
    my_df['Day'] = adgprs_df['Day']
    my_df[well_name] = 0
    col = well_name.upper() + 'BHP'
    if col in adgprs_df.columns:
        my_df[well_name] += adgprs_df[col]

    my_df.plot(x='Day', y=well_name, style = style, color = color, linewidth = 1.5, ax=ax)
    
    return ax  
    
def plot_water_rate_adgprs(well_name, adgprs_df, style = '-.', color = '#C41E3A', ax = None):
    
    my_df = pd.DataFrame()
    my_df['Day'] = adgprs_df['Day']
    my_df[well_name] = 0
    col_p = well_name.upper() + 'WPR'
    col_i = well_name.upper() + 'WIR'
    if col_p in adgprs_df.columns:
        my_df[well_name] += adgprs_df[col_p]

    if col_i in adgprs_df.columns:
        my_df[well_name] += adgprs_df[col_i]

    my_df[well_name] = -my_df[well_name]
    my_df.plot(x='Day', y=well_name, style = style, color = color, linewidth = 1.5, ax=ax)
    
    return ax

def plot_oil_rate_adgprs(well_name, adgprs_df, style = '-.', color = '#C41E3A', ax = None):
    
    my_df = pd.DataFrame()
    my_df['Day'] = adgprs_df['Day']
    my_df[well_name] = 0
    col_p = well_name.upper() + 'OPR'
    col_i = well_name.upper() + 'OIR'
    if col_p in adgprs_df.columns:
        my_df[well_name] += adgprs_df[col_p]

    if col_i in adgprs_df.columns:
        my_df[well_name] += adgprs_df[col_i]

    my_df[well_name] = -my_df[well_name]
    my_df.plot(x='Day', y=well_name, style = style, color = color, linewidth = 1.5, ax=ax)
    
    return ax

def plot_gas_rate_adgprs(well_name, adgprs_df, style = '-.', color = '#C41E3A', ax = None):
    
    my_df = pd.DataFrame()
    my_df['Day'] = adgprs_df['Day']
    my_df[well_name] = 0
    col_p = well_name.upper() + 'GPR'
    col_i = well_name.upper() + 'GIR'
    if col_p in adgprs_df.columns:
        my_df[well_name] += adgprs_df[col_p]

    if col_i in adgprs_df.columns:
        my_df[well_name] += adgprs_df[col_i]

    my_df[well_name] = -my_df[well_name]
    my_df.plot(x='Day', y=well_name, style = style, color = color, linewidth = 1.5, ax=ax)
    
    return ax


def plot_temp_adgprs(well_name, adgprs_df, style='-.', color="#C41E3A", ax=None):
    my_df = pd.DataFrame()
    my_df['Day'] = adgprs_df['Day']
    my_df[well_name] = 0
    col = well_name.upper() + 'TEMP'
    if col in adgprs_df.columns:
        my_df[well_name] += adgprs_df[col]

    my_df.plot(x='Day', y=well_name, style=style, color=color, linewidth=1.5, ax=ax)

    return ax


def plot_bhp_vs_adgprs(well_name, darts_df, adgprs_df, ax = None):
    plot_bhp_adgprs (well_name, adgprs_df, ax)
    plot_bhp_darts (well_name, darts_df, ax)
    return ax

def plot_temp_vs_adgprs(well_name, darts_df, adgprs_df, ax = None):
    plot_temp_adgprs (well_name, adgprs_df, ax)
    plot_temp_darts (well_name, darts_df, ax)
    return ax

def plot_oil_rate_vs_adgprs(well_name, darts_df, adgprs_df, ax = None):
    plot_oil_rate_adgprs (well_name, adgprs_df, ax)
    plot_oil_rate_darts (well_name, darts_df, ax)
    return ax

def plot_gas_rate_vs_adgprs(well_name, darts_df, adgprs_df, ax = None):
    plot_gas_rate_adgprs (well_name, adgprs_df, ax)
    plot_gas_rate_darts (well_name, darts_df, ax)
    return ax

def plot_water_rate_vs_adgprs(well_name, darts_df, adgprs_df, ax = None):
    plot_water_rate_adgprs (well_name, adgprs_df, ax)
    plot_water_rate_darts (well_name, darts_df, ax)
    return ax

def plot_total_prod_water_rate_adgprs(adgprs_df, style='-', color='#C41E3A', ax=None, alpha=1):
    acc_df = pd.DataFrame()
    acc_df['time'] = adgprs_df['Day']
    acc_df['total'] = 0
    search_str = '' + 'WPR'
    for col in adgprs_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += adgprs_df[col]

    acc_df['total'] = acc_df['total'].abs()
    ax = acc_df.plot(x='time', y='total', style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax

def plot_total_prod_oil_rate_adgprs(adgprs_df, style='-', color='#C41E3A', ax=None, alpha=1):
    acc_df = pd.DataFrame()
    acc_df['time'] = adgprs_df['Day']
    acc_df['total'] = 0
    search_str = '' + 'OPR'
    for col in adgprs_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += adgprs_df[col]

    acc_df['total'] = acc_df['total'].abs()
    ax = acc_df.plot(x='time', y='total', style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax

def plot_total_inj_water_rate_adgprs(adgprs_df, style='-', color='#C41E3A', ax=None, alpha=1):
    acc_df = pd.DataFrame()
    acc_df['time'] = adgprs_df['Day']
    acc_df['total'] = 0
    search_str = '' + 'WIR'
    for col in adgprs_df.columns:
        if search_str in col:
            if 'I' in col:
                for i in range(0, len(adgprs_df[col])):
                    if adgprs_df[col][i] <= 0:
                        acc_df.loc[i, 'total'] += adgprs_df[col][i]
    acc_df['total'] = -acc_df['total']
    ax = acc_df.plot(x='time', y='total', style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax
