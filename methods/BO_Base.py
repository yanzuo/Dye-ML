import os
import pickle
import random
import numpy as np
import pandas as pd
import GPy
from utils.ml.models import GP
from utils.bayesopt.acquisition import EI, UCB, PI, TS, MES, MI
from utils.ml.optimization import sample_then_minimize, sample_then_minimize_with_constraints


class BO_Base():
    """
    Base class with common operations for BO with continuous and categorical
    inputs
    """

    def __init__(self, bounds, acq_params, debug=False, ard=False, out_dims=1, **kwargs):

        # acquisition parameters
        self.acq_params = acq_params

        # bounds for function
        self.bounds = bounds

        # no: of initial points
        self.debug = debug
        self.kwargs = kwargs
        self.out_dims = out_dims

        # sampled points for surrogate
        self.X = []
        self.Y = []

        # Automatic Relevance Determination
        self.ARD = ard

        # default lengthscale for kernel
        self.default_cont_lengthscale = 0.2

        # ml hyperparameters
        self.model_update_interval = 1
        self.model_hp = None

        # list of best runs
        self.best_val_list = []

        self.name = 'BO_Base'


    def get_kernel(self, continuous_dims):
        # create surrogate
        if self.ARD:
            hp_bounds = np.array(
                [*[[1e-4, 3]] * len(continuous_dims),  # cont lengthscale
                [1e-6, 1],  # likelihood variance
            ])
        else:
            hp_bounds = np.array([
                [1e-4, 3],  # cont lengthscale
                [1e-6, 1],  # likelihood variance
            ])

        my_kernel = GPy.kern.Matern52(input_dim=len(continuous_dims),
                                      lengthscale=self.default_cont_lengthscale,
                                      active_dims=continuous_dims,
                                      ARD=self.ARD)
        my_kernel.unlink_parameter(my_kernel.variance)
        return my_kernel, hp_bounds


    def set_model_params_and_opt_flag(self, model, model_update_interval=1):
        """
        Returns opt_flag, ml
        """
        if ((self.iter >= 0) and (self.iter % model_update_interval == 0)):
            return True, model
        else:
            self.model_hp = model.param_array
            return False, model


    def update_surrogate(self, kernel, hp_bounds, model_update_interval=1):
        # set up surrogate params
        gp_opt_params = {'method': 'multigrad',
                         'num_restarts': 5,
                         'restart_bounds': hp_bounds,
                         'hp_bounds': hp_bounds,
                         'verbose': False}
        gp_kwargs = {'y_norm': 'meanstd', 'opt_params': gp_opt_params}

        Xt = self.data[0]
        if self.result[0].shape[-1] > 1:
            Yt = np.sum(self.result[0], axis=-1, keepdims=True)
        else:
            Yt = self.result[0]

        gp_args = (Xt, Yt, kernel)
        self.gp = GP(*gp_args, **gp_kwargs)
        opt_flag, self.gp = self.set_model_params_and_opt_flag(self.gp, model_update_interval)

        # fit surrogate to current points
        if opt_flag:
            # print("\noptimising!\n")
            self.gp.optimize()
        self.model_hp = self.gp.param_array

        return gp_args, gp_kwargs


    def get_acq_func(self, acq_params):

        # get parameters of acquistion function
        acq_type = acq_params['acq_type']

        best_val = np.min(self.gp.Y_raw)
        # create acq
        if acq_type == 'EI':
            acq = EI(self.gp, best_val)
        elif acq_type == 'LCB':
            kappa = acq_params['kappa']
            acq = UCB(self.gp, kappa)
        elif acq_type == 'PI':
            kappa = acq_params['kappa']
            acq = PI(self.gp, best_val, kappa)
        elif acq_type == 'TS':
            acq = TS(self.gp)
        elif acq_type == 'MES':
            acq = MES(self.gp, best_val)
        elif acq_type == 'MI':
            acq = MI(self.gp)

        def acq_func(x):
            return -acq.evaluate(np.atleast_2d(x))

        return acq_func


    # returns the reward for each MAB after BO
    def next_x(self, acq_func, num_samples=5000, num_chunks=10, num_local=3):

        # define bounds on input
        x_bounds = np.array([d['domain'] for d in self.bounds if d['type'] == 'continuous'])

        # sample from acquisition and minimise
        res = sample_then_minimize(
            acq_func,
            x_bounds,
            num_samples=num_samples,
            num_chunks=num_chunks,
            num_local=num_local,
            minimize_options=None,
            evaluate_sequentially=False)

        return res


    # returns the reward for each MAB after BO
    def next_x_constrained(self, acq_func, constrained_dims, unconstrained_dims, latest_x, num_samples=5000, num_chunks=10, num_local=3):

        # define bounds on input
        x_bounds = np.array([d['domain'] for d in self.bounds if d['type'] == 'continuous'])

        # set bounds on constrained dims
        for d in constrained_dims:
            x_bounds[d, 0] = latest_x[d]
            x_bounds[d, 1] = latest_x[d]

        # sample from acquisition and minimise
        res = sample_then_minimize_with_constraints(
            acq_func,
            x_bounds,
            unconstrained_dims,
            latest_x[constrained_dims],
            num_samples=num_samples,
            num_chunks=num_chunks,
            num_local=num_local,
            minimize_options=None,
            evaluate_sequentially=False)

        return res


    # gives recommendation given data and surrogate
    def suggest(self, constrained_dims=None, n_suggestions=1, init_x=None, init_y=None):

        self.iter = 0

        # create initial data samples
        self.data = []
        self.result = []
        self.data.append(init_x[:])
        self.result.append(init_y[:])

        # number of dimensions in continuous inputs
        n_dim = len(self.bounds)
        continuous_dims = list(range(n_dim))

        # get kernel for BO
        kernel, hp_bounds = self.get_kernel(continuous_dims)

        # update the surrogate ml
        self.update_surrogate(kernel, hp_bounds, model_update_interval=self.model_update_interval)

        # get acquisition function
        acq_func = self.get_acq_func(acq_params=self.acq_params)

        # get next query point
        latest_x = self.data[0][-1]
        if constrained_dims:
            unconstrained_dims = list(set(continuous_dims) - set(constrained_dims))
            res = self.next_x_constrained(acq_func, constrained_dims, unconstrained_dims, latest_x)
        else:
            res = self.next_x(acq_func)

        return res