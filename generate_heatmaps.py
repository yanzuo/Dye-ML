import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from methods.BO_Base import BO_Base

# get arguments
parser = argparse.ArgumentParser(description="Generate Surrogate Heatmaps for Dye Experiments")
parser.add_argument('-n', '--name', help='Name used for save directory for heatmaps', default=None, type=str)
parser.add_argument('-f', '--filepath', help='Filepath of file containing experiments', default='datasets/dye_bo_doe.xlsx', type=str)
parser.add_argument('-i','--intervals', help='List of time intervals for generating heatmaps', default=None, nargs='+')
args = parser.parse_args()
print(f"Got arguments: \n{args}")


if __name__ == '__main__':

    # specify bounds and output for objective
    bounds = [
        {'name': 'dye_processed', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'flowrate', 'type': 'continuous', 'domain': (0.01, 1)},
        {'name': 'temp', 'type': 'continuous', 'domain': (0.25, 1)}]
    out_dims = 1

    # define variables of interest
    var_names = np.array([
        'Dye Since Last Regen (mg)',
        'Set Pump Rate (ml/min)',
        'TSET (°C)',
        'Regenerated Catalyst',
        'Conversion per minute (%/min)'
    ])

    # set up csv file to read in initial data points
    data = pd.read_excel(args.filepath,
                         sheet_name='Summary',
                         header=1,
                         index_col=None,
                         usecols=var_names)

    # filtering out any invalid entries in data
    data = data.dropna()

    # normalise data for optimisation
    data['Dye Since Last Regen (mg)'] /= 100
    data['Set Pump Rate (ml/min)'] /= 10.0
    data['TSET (°C)'] /= 80.0
    data['Conversion per minute (%/min)'] /= -100.0

    # extract data from dataframe
    data = data[[
        'Dye Since Last Regen (mg)',
        'Set Pump Rate (ml/min)',
        'TSET (°C)',
        'Conversion per minute (%/min)'
    ]]

    # create initial x and y data
    data_np = data.to_numpy()

    # use specified intervals or generate all heatmaps
    if args.intervals is None:
        interval_range = range(1, len(data_np))
    else:
        interval_range = args.intervals
    # convert interval range into range of ints
    interval_range = [int(i) for i in interval_range]

    # define saving path for saving the results
    if args.name:
        save_directory = f'data/runs/{args.name}/'
    else:
        save_directory = f'data/runs/dye_bo/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # for visualising heatmap evolution
    for num in interval_range:

        # for generating heatmaps, take a subset of data up to index
        sub_data = data_np[:num, :]

        # load in x and y for surrogate
        init_x = sub_data[:, :len(bounds)]
        init_y = sub_data[:, -out_dims:]

        # load in appropriate model for pure continuous or mixed input
        acq_params = {'acq_type': 'EI'}
        model = BO_Base(bounds=bounds, acq_params=acq_params, out_dims=out_dims)

        # create initial data samples
        model.iter = num
        model.data = []
        model.result = []
        model.data.append(init_x[:])
        model.result.append(init_y[:])

        # get kernel for BO
        # number of dimensions in continuous inputs
        n_dim = len(model.bounds)
        continuous_dims = list(range(n_dim))
        kernel, hp_bounds = model.get_kernel(continuous_dims)

        # update the surrogate ml
        model.update_surrogate(kernel, hp_bounds, model_update_interval=model.model_update_interval)

        # create 2d map of values
        x1 = np.arange(0, 1, 0.025)
        x2 = np.arange(0, 1, 0.02)
        grid_map = []
        for i in x1:
            row = []
            for j in x2:
                row.append([0, i, j])
            grid_map.append(row)
        grid_map = np.array(grid_map)
        grid_mean = []
        grid_std = []
        for i in range(40):
            row_mean = []
            row_std = []
            for j in range(50):
                pred_mean, pred_std = model.gp.predict(grid_map[i,j,:][None,:])
                row_mean.append(pred_mean)
                row_std.append(pred_std)
            grid_mean.append(row_mean)
            grid_std.append(row_std)
        grid_mean = -np.squeeze(np.array(grid_mean))
        grid_std = np.squeeze(np.array(grid_std))

        # plot heat map
        ax = sns.heatmap(grid_mean, xticklabels=5, yticklabels=5, cbar=False, vmin=0, vmax=1.0)
        ax.invert_yaxis()
        temps = np.arange(0, 80, 10)
        flow_rates = np.arange(0, 10, 1)
        ax.set_xticklabels(flow_rates)
        ax.set_yticklabels(temps)
        plt.rcParams.update({'font.size': 14})
        plt.xlabel('Set Pump Rate (ml/min)')
        plt.ylabel('Temperature (°C)')
        # plt.show()

        # save as png
        save_filepath = os.path.join(save_directory, f'dye_heat_map_{num}.png')
        plt.savefig(save_filepath)

        plt.clf()






