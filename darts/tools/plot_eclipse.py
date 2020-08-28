import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_bhp_eclipse(well_name, eclipse_df, ax = None):

    my_df = pd.DataFrame()
    my_df['DAYS'] = eclipse_df['TIME']
    my_df[well_name] = 0
    col = well_name.upper() + 'WBHP'
    if col in eclipse_df.columns:
        my_df[well_name] += eclipse_df[col]

    ax = my_df.plot(x='DAYS', y=well_name, style = '--', color = 'r', linewidth = 1.5, label = 'ECLIPSE', ax=ax)

    plt.show(block=False)
    return ax

def plot_block_pressure_eclipse(well_name, eclipse_df, ax = None):

    my_df = pd.DataFrame()
    my_df['DAYS'] = eclipse_df['TIME']
    my_df[well_name] = 0
    col = well_name.upper() + 'BPR'
    if col in eclipse_df.columns:
        my_df[well_name] += eclipse_df[col]

    my_df.plot(x='DAYS', y=well_name, style = '--', color = 'r', linewidth = 1.5, label = 'ECLIPSE', ax=ax)

    plt.show(block=False)
    return ax


def plot_oil_rate_eclipse(well_name, eclipse_df, ax = None):
    my_df = pd.DataFrame()
    my_df['DAYS'] = eclipse_df['TIME']
    my_df[well_name] = 0
    col_p = well_name.upper() + 'FOPR'
    col_i = well_name.upper() + 'FOIR'
    if col_p in eclipse_df.columns:
        my_df[well_name] += eclipse_df[col_p]

    if col_i in eclipse_df.columns:
        my_df[well_name] += eclipse_df[col_i]
        my_df[well_name] = -my_df[well_name]

    my_df[well_name] = -my_df[well_name]
    ax = my_df.plot(x='DAYS', y=well_name, style = '--', color = 'r', linewidth = 1.5, label = 'ECLIPSE', ax=ax)

    plt.show(block=False)
    return ax

def plot_water_rate_eclipse(well_name, eclipse_df, ax = None):
    my_df = pd.DataFrame()
    my_df['DAYS'] = eclipse_df['TIME']
    my_df[well_name] = 0
    col_p = well_name.upper() + 'FWPR'
    col_i = well_name.upper() + 'FWIR'
    if col_p in eclipse_df.columns:
        my_df[well_name] += eclipse_df[col_p]

    if col_i in eclipse_df.columns:
        my_df[well_name] += eclipse_df[col_i]
        my_df[well_name] = -my_df[well_name]

    my_df[well_name] = -my_df[well_name]
    ax = my_df.plot(x='DAYS', y=well_name, style = '--', color = 'r', linewidth = 1.5, label = 'ECLIPSE', ax=ax)

    plt.show(block=False)
    return ax

def plot_gas_rate_eclipse(well_name, eclipse_df, ax = None):
    my_df = pd.DataFrame()
    my_df['DAYS'] = eclipse_df['TIME']
    my_df[well_name] = 0
    col_p = well_name.upper() + 'FGPR'
    col_i = well_name.upper() + 'FGIR'
    if col_p in eclipse_df.columns:
        my_df[well_name] += eclipse_df[col_p]

    if col_i in eclipse_df.columns:
        my_df[well_name] += eclipse_df[col_i]
        my_df[well_name] = -my_df[well_name]

    my_df[well_name] = -my_df[well_name]
    ax = my_df.plot(x='DAYS', y=well_name, style = '--', color = 'r', linewidth = 1.5, label = 'ECLIPSE', ax=ax)

    plt.show(block=False)
    return ax