import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from methods.BO_Base import BO_Base

# get arguments
parser = argparse.ArgumentParser(description="Generate Catalyst Degradation Curve for Dye Experiments")
parser.add_argument('-n', '--name', help='Name used for save directory for heatmaps', default=None, type=str)
parser.add_argument('-f', '--filepath', help='Filepath of file containing experiments', default='datasets/dye_bo_doe.xlsx', type=str)
parser.add_argument('-i','--n_samples', help='Number of samples for generating catalyst curve', default=500, type=int)
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

    # load in x and y for surrogate
    init_x = data_np[:, :len(bounds)]
    init_y = data_np[:, -out_dims:]

    # define saving path for saving the results
    if args.name:
        save_directory = f'data/runs/{args.name}/'
    else:
        save_directory = f'data/runs/dye_bo/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # load in appropriate model for pure continuous or mixed input
    acq_params = {'acq_type': 'EI'}
    model = BO_Base(bounds=bounds, acq_params=acq_params, out_dims=out_dims)

    # create initial data samples
    model.iter = 0
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

    # create catalyst degradation curve
    n_samples = args.n_samples
    x = np.ones(shape=(n_samples,3))

    # find input that gave the best observed outcome
    flowrate_dim = 1
    temp_dim = 2
    best_x = data_np[np.argmin(data_np[:,-1])]
    # input that gave best observed outcome
    x[:,flowrate_dim] *= best_x[flowrate_dim]
    x[:,temp_dim] *= best_x[temp_dim]
    # run up from 0 to 70mg of dye flowed
    x[:, 0] = np.random.uniform(0, 0.65, n_samples)
    x = np.sort(x, axis=0)
    # get surrogate prediction on x
    pred_mean, pred_std = model.gp.predict(x)
    # normalise prediction mean to efficacy
    pred_mean /= np.min(pred_mean)
    curve_data = np.concatenate([x[:,:1]*100, pred_mean, pred_std], axis=-1)
    curve_df = pd.DataFrame(columns=['Dye over catalyst (mg)', 'Efficacy (%)', 'Efficacy Variance (%)'], data=curve_data)
    ax = sns.lineplot(x='Dye over catalyst (mg)', y='Efficacy (%)', data=curve_df, markers=True)
    ax.fill_between(curve_df['Dye over catalyst (mg)'],
                    y1=curve_df['Efficacy (%)'] - 2*curve_df['Efficacy Variance (%)'],
                    y2=curve_df['Efficacy (%)'] + 2*curve_df['Efficacy Variance (%)'], alpha=.25)
    plt.xlabel('Dye over catalyst (mg)')
    plt.ylabel('Efficacy (%)')
    plt.xticks(np.arange(0, 65, 6))
    # plt.show()

    save_filepath = os.path.join(save_directory, f'catalyst_degradation_curve.png')
    plt.savefig(save_filepath)



