import argparse
import numpy as np
import pandas as pd
from methods.BO_Base import BO_Base

# get arguments
parser = argparse.ArgumentParser(description="Run BayesOpt Experiments for Photocatalytic Dye Trials")
parser.add_argument('-f', '--filepath', help='Filepath of file containing experiments', default='datasets/dye_bo_doe.xlsx', type=str)
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

    # load in appropriate model for pure continuous or mixed input
    acq_params = {'acq_type': 'EI'}
    model = BO_Base(bounds=bounds, acq_params=acq_params, out_dims=out_dims)

    # get single recommendation from model given data
    res = model.suggest(constrained_dims=[0], init_x=init_x, init_y=init_y)

    # convert result back to actual experimental values
    df = pd.DataFrame(res.x.reshape(1, -1), columns=['dye_processed', 'flowrate', 'temp'])
    dye_processed = df['dye_processed'].iloc[0] * 100.0
    flowrate = df['flowrate'].iloc[0] * 10.0
    temp = df['temp'].iloc[0] * 80.0
    print(f' Next recommended configuration:')
    print(f' Temperature (°C) = {temp}; Set Flow Rate (ml/min) = {flowrate}')



