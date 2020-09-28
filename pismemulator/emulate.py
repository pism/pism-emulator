# Copyright (C) 2019 Rachel Chen, Andy Aschwanden
#
# This file is part of pism-emulator.
#
# PISM-EMULATOR is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PISM-EMULATOR is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

# emulate.py contains functions to generate kernels and perform regression analysis.

import GPy as gp
import numpy as np

from .utils import prepare_data, stepwise_bic


def generate_kernel(varlist, kernel, varnames):
    """
    Generate a kernel based on a list of model terms

    Currently only supports GPy but could be easily extended
    to gpflow.

    :param varlist: list of strings containing model terms.
    :param kernel: GPy.kern instance
    :param var_names: list of strings containing variable names

    :return kernel: GPy.kern instance

    Example:

    To generate a kernel that represents the model

    Y ~ X1 + X2 + X1*X2 with an exponential kernel

    use

    varlist = ["X1", "X2", "X1*X2"]
    kernel = GPy.kern.Exponential
    varnames = ["X1", "X2"]

    """

    params_dict = {k: v for v, k in enumerate(varnames)}

    first_order = [params_dict[x] for x in varlist if "*" not in x]
    interactions = [x.split("*") for x in varlist if "*" in x]
    nfirst = len(first_order)

<<<<<<< Updated upstream
    kern = kernel(input_dim=nfirst, active_dims=first_order, ARD=True)
=======
    ls = 25
    kern = kernel(input_dim=nfirst, active_dims=first_order, ls=np.ones(nfirst) * ls)
>>>>>>> Stashed changes

    mult = []
    active_intx = []
    for i in range(len(interactions)):
        mult.append([])
        for j in list(range(len(interactions[i]))):
            active_dims = [params_dict[interactions[i][j]]]
            mult[i].append(kernel(input_dim=1, active_dims=active_dims, ARD=True))
        active_intx.append(np.prod(mult[i]))
    for k in active_intx:
        kern += k

    return kern


<<<<<<< Updated upstream
def emulate_gp(
    samples, response, X_new, kernel=gp.kern.Exponential, stepwise=False, optimizer_options={}, regressor_options={}
):
=======
def emulate_gp(samples, response, X_new, kernel=pm.gp.cov.Exponential, stepwise=False):
>>>>>>> Stashed changes

    """
    Perform Gaussian Process emulation

    :param samples: pandas.DataFrame instance of samples
    :param response: pandas.DataFrame instance of response
    :param X_new: numpy.ndarray
    :param kernel: GPy.kern instance
    :param stepwise: use stepwiseBIC to generate kernel
    :param optimizer_options: dictionary with options to be passed
     on to the optimizer
    """
    X = samples.values
    Y = response.values
    n = X.shape[1]
    varnames = samples.columns

<<<<<<< Updated upstream
    if stepwise:
        steplist = stepwise_bic(X, Y, varnames=varnames)
        kern = generate_kernel(steplist, kernel=kernel, varnames=varnames)
    else:
        kern = kernel(input_dim=n, ARD=True)
=======
    with pm.Model() as model:
        ls = pm.HalfCauchy("ls", shape=n, beta=25)
        if stepwise:
            steplist = stepwise_bic(X, Y, varnames=varnames)
            kern = generate_kernel(steplist, kernel=kernel, varnames=varnames)
        else:
            kern = kernel(input_dim=n, ls=ls)
>>>>>>> Stashed changes

    m = gp.models.GPRegression(X, Y, kern, **regressor_options)
    f = m.optimize(messages=True, **optimizer_options)

    try:
        p = m.predict(X_new.values)
    except:
        p = m.predict(X_new)

    # Instead of f.status we could also return f or a bool converged.
    return p, f.status


def emulate_sklearn(samples, response, X_new, method="lasso", alphas=np.linspace(1e-4, 10, 10001), return_stats=False):

    """
    Perform emulation using sklearn

    :param samples: pandas.DataFrame instance of samples
    :param response: pandas.DataFrame instance of response
    :param X_new: numpy.ndarray
    :param method: use one of the following regression methods
      "lasso": use the LASSO method (add citation)
      "lasso-lars": use the LASSO method and model BIC model selection(add citation)
      "ridge": use the Ridge method (add citation)
    """

    supported_methods = ("lasso", "lasso-lars", "ridge")

    if method == "lasso":
        from sklearn.linear_model import LassoCV as Regressor

        regressor = Regressor(alphas=alphas)
    elif method == "lasso-lars":
        from sklearn.linear_model import LassoLarsIC as Regressor

        regressor = Regressor(criterion="bic")
    elif method == "ridge":
        from sklearn.linear_model import RidgeCV as Regressor

        regressor = Regressor(alphas=alphas)
    else:
        print(
            "Method {} not supported, supported methods are {}".format(
                method, ", ".join([m for m in supported_method])
            )
        )

    X = samples.values
    Y = response.values.ravel()

    regressor.fit(X, Y)

    model_stats = {"rsq": regressor.score(X, Y), "coefs": regressor.coef_, "alpha": regressor.alpha_}

    try:
        p = regressor.predict(X_new.values)
    except:
        p = regressor.predict(X_new)

    if return_stats:
        return p, model_stats
    else:
        return p


def gp_loo_mp(loo_idx, samples, response, kernel, stepwise):

    """
    Perform Leave-One-Out validation

    Can be called using multiprocessing

    :param loo_idx: indices to be removed from samples and response
    :param samples: pandas.DataFrame instance of samples
    :param response: pandas.DataFrame instance of response
    :param kernel: GPy.kern instance
    :param stepwise: use stepwiseBIC to generate kernel

    :return Y_p_i, Y_var_i, (p_i - Y_p_i) ** 2 / Y_var_i

    FIXME explain return better

    Example

    from functools import partial
    import GPy as gp
    from multiprocessing import Pool

    with Pool(n_procs) as pool:

        results = pool.map(
            partial(gp_loo_mp, samples=samples, response=response, kernel=gp.kern.Exponential, stepwise=False),
                    range(len(response)),
                )
        pool.close()

    """
    X_p = samples.loc[loo_idx].values.reshape(1, -1)
    Y_p = response.loc[loo_idx].values

    p, converged = emulate_gp(
        samples.drop(loo_idx),
        response.drop(loo_idx),
        X_p,
        kernel=kernel,
        stepwise=stepwise,
    )

    if converged:
        return np.squeeze(Y_p), np.squeeze(p[0]), np.squeeze(p[1]), np.squeeze((p[0] - Y_p) ** 2 / p[1]), loo_idx


def gp_response_mp(response_file, samples_file, X_p, kernel, stepwise):

    """
    Perform emulation

    Can be called using multiprocessing

    :param loo_idx: indices to be removed from samples and response
    :param samples: pandas.DataFrame instance of samples
    :param response: pandas.DataFrame instance of response
    :param kernel: GPy.kern instance
    :param stepwise: use stepwiseBIC to generate kernel

    :return Y_p_i, Y_var_i, (p_i - Y_p_i) ** 2 / Y_var_i

    FIXME explain return better

    Example

    from functools import partial
    import GPy as gp
    from multiprocessing import Pool

    with Pool(n_procs) as pool:

        results = pool.map(
            partial(gp_loo_mp, samples=samples, response=response, kernel=gp.kern.Exponential, stepwise=False),
                    range(len(response)),
                )
        pool.close()

    """

    samples, response = prepare_data(samples_file, response_file)

    p, converged = emulate_gp(
        samples,
        response,
        X_p,
        kernel=kernel,
        stepwise=stepwise,
        optimizer_options={"max_iters": 4000},
        regressor_options=regressor_options,
    )

    if converged:
        return np.squeeze(p[0]), np.squeeze(p[1]), response_file
