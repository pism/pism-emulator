#!/env/bin python
#
# This script tests different Gaussian Process kernels using a
# Leave-One-Out methods as described in Edwards et al. (2019)

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import pandas as pd
import os

def read_data(year, rcp, samples_file, return_samples = False):
    response_file = os.path.join("response", "dgmsl_rcp_{}_year_{}.csv").format(rcp, year)

    print("\nProcessing {}".format(response_file))

    # Load Samples file as Pandas DataFrame
    samples = pd.read_csv(samples_file, delimiter=",", squeeze=True, skipinitialspace=True)

    # Load Response file as Pandas DataFrame
    response = pd.read_csv(response_file, delimiter=",", squeeze=True, skipinitialspace=True)
    # It is possible that not all ensemble simulations succeeded and returned a value
    # so we much search for missing response values
    missing_ids = list(set(samples["id"]).difference(response["id"]))
    Y = response[response.columns[-1]].values.reshape(1, -1).T
    if missing_ids:
        print("The following simulation ids are missing:\n   {}".format(missing_ids))
        # and remove the missing samples
        samples_missing_removed = samples[~samples["id"].isin(missing_ids)]
        X = samples_missing_removed.values[:, 1::]

    else:
        X = samples.values[:, 1::]

    if return_samples:
        return X, Y, samples
    else:
        return X, Y


if __name__ == "__main__":

    __spec__ = None

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Gaussian Process Emulator: Leave-One-Out analysis to test different kernels."
    parser.add_argument(
        "-n", "--n_procs", dest="n_procs", type=int, help="""number of cores/processors. default=4.""", default=4
    )
    parser.add_argument(
        "-a", "--start_year", dest="start_year", type=int, help="""Start year. default=2010.""", default=2010
    )
    parser.add_argument(
        "-e", "--end_year", dest="end_year", type=int, help="""End year. default=2300.""", default=2300
    )
    parser.add_argument("--rcp", help="RCP scenario. Default=85", default="85")
    parser.add_argument(
        "-s",
        "--samples_file",
        dest="samples_file",
        help="File that has all combinations for ensemble study",
        default="samples/samples.csv",
    )

    options = parser.parse_args()
    n_procs = options.n_procs
    rcp = options.rcp
    start_year = options.start_year
    end_year = options.end_year
    samples_file = options.samples_file

    directories = ["emulate", "emulate_stepwise", "lasso", "lasso_lars_ic", "ridge"]

    odir = os.path.join("analyze")
    if not os.path.isdir(odir):
        os.makedirs(odir)

    master_perc = []
    error_count = 0
    missing_yrs_count = 0
    master_missing = {"26": [], "45": [], "85": []}
    missing_yrs = {"emulate": [], "emulate_stepwise": [], "lasso": [], "lasso_lars_ic": [], "ridge": []}
    errors = {"emulate": [], "emulate_stepwise": [], "lasso": [], "lasso_lars_ic": [], "ridge": []}
    for year in range(start_year, end_year + 1):
        dfs = []
        percentiles = []
        for d in directories:
            filepath = os.path.join(".", "gp", d, "gp_rcp_{}_{}.csv".format(rcp, year))
            if os.path.isfile(filepath):
                data = pd.read_csv(filepath)
                dfs.append(data)

                prct = np.percentile(data["prediction"], [5, 16, 50, 84, 95])
                print("RCP {} {} {}: {}".format(rcp, year, d, prct))
                percentiles.append(np.append((year, d), prct))
                if np.all(prct == 0):# or np.any(prct == NaN):
                    errors[d].append(year)
                    error_count += 1
            else:
                missing_yrs[d].append(year)
                missing_yrs_count += 1

        master_perc.append(np.vstack(percentiles))

    master_missing[rcp] = missing_yrs

    if error_count > 0:
        print("The following years for RCP {} method {} have errors (zeros for predictions or NaN):\n      {}".format(rcp, d, errors))

    if missing_yrs_count > 0:
        print("The following years for RCP {} are missing:\n      {}".format(rcp, missing_yrs))

    if error_count == 0:
        perc_dfs = []
        row_ind = []
        true_dfs = []
        for year in range(start_year, end_year + 1):
            if year in master_missing[rcp]:
                continue
            X, Y = read_data(year, rcp, samples_file)
            true_prct = np.percentile(Y, [5, 16, 50, 84, 95])
            true_dat = pd.DataFrame(np.append((year, "PISM"), true_prct)).T
            true_dat.columns = ["year", "method", "5", "16", "50", "84", "95"]
            true_dfs.append(true_dat)
        for m in range(len(master_perc)):
            full_dat = np.vstack((true_dfs[m], master_perc[m]))
            p_df = pd.DataFrame(full_dat, columns = ["year", "method", "5", "16", "50", "84", "95"])
            perc_dfs.append(p_df)
    
        master_df = pd.concat(perc_dfs)
        master_df.to_csv(os.path.join(odir, "analyze_gp_rcp_{}.csv".format(rcp)))
        print(master_df)

        #dirpath = os.path.join(".", rcp)
        #for year in range(start_year, end_year + 1):
        #    for t in true_dfs:
        #        print(year)
        #        print(t)
        #        t.to_csv(os.path.join(dirpath, "analyze_gp_{}_rcp_{}.csv".format(year, rcp)))            
