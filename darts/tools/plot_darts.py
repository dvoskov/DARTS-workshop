import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_bhp_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None):
    search_str = well_name + ' : BHP'
    ax = darts_df.plot(x='time', y=[col for col in darts_df.columns if search_str in col], style=style, color=color,
                       ax=ax)

    plt.show(block=False)
    return ax


def plot_oil_rate_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    search_str = well_name + ' : oil rate'
    ax = darts_df.plot(x='time', y=[col for col in darts_df.columns if search_str in col], style=style, color=color,
                       ax=ax, alpha=alpha)

    plt.show(block=False)
    return ax

def plot_oil_rate_darts_2(well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    search_str = well_name + ' : oil rate'
    darts_df['time']  = -darts_df['time']
    ax = darts_df.plot(x='time', y=[col for col in darts_df.columns if search_str in col], style=style, color=color,
                       ax=ax, alpha=alpha)

    plt.show(block=False)
    return ax

def plot_gas_rate_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None):
    search_str = well_name + ' : gas rate'
    ax = darts_df.plot(x='time', y=[col for col in darts_df.columns if search_str in col], style=style, color=color,
                       ax=ax)

    plt.show(block=False)
    return ax


def plot_water_rate_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    search_str = well_name + ' : water rate'
    ax = darts_df.plot(x='time', y=[col for col in darts_df.columns if search_str in col], style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax


def plot_watercut_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1, label=''):
    wat = well_name + ' : water rate (m3/day)'
    oil = well_name + ' : oil rate (m3/day)'
    wcut = well_name + ' watercut'
    darts_df[wcut] = darts_df[wat] / (darts_df[wat] + darts_df[oil])

    if label == '':
        label = wcut
    ax = darts_df.plot(x='time', y=wcut, style=style, color=color,
                       ax=ax, alpha=alpha, label=label)

    ax.set_ylim(0, 1)


    plt.show(block=False)
    return ax

def plot_water_rate_darts_2(well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    search_str = well_name + ' : water rate'
    darts_df['time']  = -darts_df['time']
    ax = darts_df.plot(x='time', y=[col for col in darts_df.columns if search_str in col], style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax

def plot_water_rate_vs_obsrate(well_name, darts_df, truth_df,  style='-', color='#00A6D6', ax=None, marker="o"):
    search_str = well_name + ' : water rate'
    darts_df = darts_df.set_index('time', drop=False)
    ax.scatter(x=abs(darts_df[[col for col in truth_df.columns if search_str in col]].loc[truth_df.time, :]),
               y=abs(truth_df[[col for col in truth_df.columns if search_str in col]]), marker=marker)
    max_rate = max(max(abs(darts_df[search_str + ' (m3/day)'].values)), max(abs(truth_df[search_str + ' (m3/day)'].values)))
    min_rate = min(min(abs(darts_df[search_str + ' (m3/day)'].values)), min(abs(truth_df[search_str + ' (m3/day)'].values)))
    plt.plot( [max_rate, min_rate], [max_rate, min_rate], color=color, marker='x')
    plt.ylabel('Truth Data')
    plt.xlabel('Simulation Data')
    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)
    plt.grid(True)
    plt.show(block=False)
    return ax

def plot_water_rate_vs_obsrate_time(well_name, darts_df, truth_df,  style='-', color='#00A6D6', ax=None, marker="o", time=0):
    search_str = well_name + ' : water rate'
    darts_df = darts_df.set_index('time', drop=False)
    truth_df = truth_df.set_index('time', drop=False)
    ax.scatter(x=abs(darts_df[[col for col in truth_df.columns if search_str in col]].loc[truth_df.time[time], :]),
               y=abs(truth_df[[col for col in truth_df.columns if search_str in col]].loc[truth_df.time[time], :]), marker=marker, label=(well_name+' time: '+ time.__str__()))
    ax.legend(loc=5)
    plt.ylabel('Truth Data')
    plt.xlabel('Simulation Data')
    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)
    plt.grid(True)
    plt.show(block=False)
    return ax

def plot_oil_rate_rate_vs_obsrate(well_name, darts_df, truth_df,  style='-', color='#00A6D6', ax=None, marker="o"):
    search_str = well_name + ' : oil rate'
    darts_df = darts_df.set_index('time', drop=False)
    ax.scatter(x=abs(darts_df[[col for col in truth_df.columns if search_str in col]].loc[truth_df.time, :]),
               y=abs(truth_df[[col for col in truth_df.columns if search_str in col]]), marker=marker)
    min_oil_rate_sim = min(abs(darts_df[search_str + ' (m3/day)'].values))
    min_oil_rate_truth = min(abs(truth_df[search_str + ' (m3/day)'].values))
    max_oil_rate_sim = max(abs(darts_df[search_str + ' (m3/day)'].values))
    max_oil_rate_truth = max(abs(truth_df[search_str + ' (m3/day)'].values))
    max_rate = max(max_oil_rate_sim, max_oil_rate_truth)
    min_rate = min(min_oil_rate_sim, min_oil_rate_truth)
    plt.plot( [max_oil_rate_truth, min_oil_rate_truth], [max_oil_rate_truth, min_oil_rate_truth], color=color, marker='x')
    plt.ylabel('Truth Data')
    plt.xlabel('Simulation Data')

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)
    plt.grid(True)
    plt.show(block=False)
    return ax

def plot_total_inj_water_rate_darts(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : water rate'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) > 0:
            if 'I' in col:
            #     acc_df['total'] += darts_df[col]
                for i in range(0, len(darts_df[col])):
                    if darts_df[col][i] >= 0:
                        # acc_df['total'][i] += darts_df[col][i]
                        acc_df.loc[i, 'total'] += darts_df[col][i]

    ax = acc_df.plot(x='time', y='total', style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax

def plot_total_inj_gas_rate_darts(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : gas rate'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) > 0:
            if 'I' in col:
            #     acc_df['total'] += darts_df[col]
                for i in range(0, len(darts_df[col])):
                    if darts_df[col][i] >= 0:
                        # acc_df['total'][i] += darts_df[col][i]
                        acc_df.loc[i, 'total'] += darts_df[col][i]

    ax = acc_df.plot(x='time', y='total', style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax

def plot_water_rate_prediction(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : water rate'
    for col in darts_df.columns:
        if search_str in col:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

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


def plot_total_prod_water_rate_darts(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : water rate'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

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

def plot_acc_prod_water_rate_darts(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = 'water  acc'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

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

def plot_acc_prod_oil_rate_darts(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = 'oil  acc'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

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

def plot_total_prod_oil_rate_darts(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : oil rate'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

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

def plot_total_prod_gas_rate_darts(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : gas rate'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

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

def plot_temp_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None):
    search_str = well_name + ' : temperature'
    ax = darts_df.plot(x='time', y=[col for col in darts_df.columns if search_str in col], style=style, color=color,
                       ax=ax)

    plt.show(block=False)
    return ax
