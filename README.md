# Interchange Bayesian Optimisation for Azo-Dye Decomposition Experiments
This repository contains the framework for running the Interchange Azo-Dye Decomposition Project. 
This framework is designed to be used in a UNIX environment. If you are using a Windows machine, it is recommended to
first set up Windows Subsystem Linux. For more information, please refer to [here](https://docs.microsoft.com/en-us/windows/wsl/install).

## Main dependencies:
* python3
* numpy
* scipy
* gpy
* pandas
* h5py

For a full list of dependencies (including optional dependencies), see the `requirements.yml` file

## Setup:

### Installing Anaconda
Conda needs to be installed and updated to the latest stable version. For installing Conda, first download the navigator.
Refer to archive listing for the version that you wish to install [here](https://repo.anaconda.com/archive/).
For example to download the navigator for the version `Anaconda3-2021.05-Linux-x86_64`:
```
$ wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
```

After the Conda installer script is downloaded, it needs to be installed:
```
$ ./Anaconda3-2021.05-Linux-x86_64.sh
```

### Updating Anaconda
To update Conda to the latest stable version:
```
$ conda update conda
$ conda update --all
```

### Creating Anaconda Environment
From the root directory of this repository, to create the Conda environment to run code from this repository:
```
$ conda config --set channel_priority strict
$ conda env create -f requirements.yml
```
This should set up the conda environment with all prerequisites for running this code. Activate this Conda
environment using the following command:
```
$ conda activate dyebo
```

## Usage:
To run experiments: `python run_dye_trial.py` followed by the following flags:
* `-f` Experimental data filepath: default=`'datasets/dye_bo_doe.xlsx'`.

For example: `python run_dye_trial.py -f='datasets/dye_bo_doe.xlsx'`

## Generating surrogate heat maps:
To generate best value lineplots: `python generate_heatmaps.py` followed by the following flags:
* `--name` Name used for save directory for heatmaps. If not specified `'dye_bo'` will be used: default=`None`
* `-f` Filepath for experimental data: default=`'datasets/dye_bo_doe.xlsx'`
* `-i` Sampling interval for generating heatmaps. If not specified, heatmap will be generated for each experiment: default=`None`
* 
For example: `python generate_heatmaps.py --name='dye_bo' -f='datasets/dye_bo_doe.xlsx' -i=2 4 6 8`

## Generating catalyst degradation curve:
To generate best value lineplots: `python generate_catalyst_curve.py` followed by the following flags:
* `--name` Name used for save directory for heatmaps. If not specified `'dye_bo'` will be used: default=`None`
* `-f` Filepath for experimental data: default=`'datasets/dye_bo_doe.xlsx'`
* `-i` Number of points to sample for generating catalyst degradation curve: default=`500`

For example: `python generate_catalyst_curve.py --name='dye_bo' -f='datasets/dye_bo_doe.xlsx' -i=500`
