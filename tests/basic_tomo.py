import numpy as np
import pandas as pd
from mats_l2_processing.forward_model import calc_jacobian
from mats_utils.rawdata.read_data import read_MATS_data
import mats_l2_processing.inverse_model as mats_inv
import pickle
import datetime as DT
import argparse
import os
import scipy.sparse as sp
from mats_l2_processing.grids import sph2cart, geoid_radius

EARTH_RADIUS = 6371000


def get_args():
    parser = argparse.ArgumentParser(description="Test 2D tomography",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=str, default="verdec2d.pickle",
                        help="Pickle filename for L1b data. Fetch data from aws and pickle if not found.")
    parser.add_argument("--jacobian", type=str, default="jacobian_3.pkl",
                        help="Pickle filename for Jacobian data. Calculate if not found.")
    parser.add_argument("--out", type=str, default="L2_tomo.pkl",
                        help="File to save the results to.")
    parser.add_argument("--solver", type=str, choices=["cgm", "spsolve"], default="cgm",
                        help="File to save the results to.")
    parser.add_argument("--start_time", type=int, nargs=5, default=[2023, 3, 31, 21, 0],
                        help="Start time for data set.")
    parser.add_argument("--stop_time", type=int, nargs=5, default=[2023, 3, 31, 22, 35],
                        help="Start time for data set.")
    parser.add_argument("--nprofiles", type=int, default=50, help="Number of profiles to get.")

    parser.add_argument("--apr_peak_val", type=float, default=3.2e12, help="A priori VER value at peak")
    parser.add_argument("--apr_peak_alt", type=float, default=93000, help="A priori VER peak altitude.")
    parser.add_argument("--apr_peak_width", type=float, default=3700, help="A priori VER peak width.")
    parser.add_argument("--Sa_weights", type=float, nargs=4, default=[1, 500, 20000, 20000],
                        help="Tikhonov regularisation weights.")
    parser.add_argument("--offset", type=int, default=300, help="Profile selection offset")
    parser.add_argument("--reg_analysis", action="store_true",
                        help="Store xa and calculate additional regularisation parameters")

    return parser.parse_args()


def get_data(args):
    dftop = read_MATS_data(DT.datetime(*args.start_time), DT.datetime(*args.stop_time), level="1b", version="0.4")

    with open(args.data, 'wb') as handle:
        pickle.dump(dftop, handle)
    return dftop


def get_jacobian(args, dftop):
    df = dftop[dftop['channel'] == 'IR2'].dropna().reset_index(drop=True)
    df = df.loc[args.offset:args.offset + args.nprofiles - 1]
    df = df.reset_index(drop=True)
    columns = np.arange(0, df["NCOL"][0], 2)
    rows = np.arange(0, df["NROW"][0] - 10, 1)

    y, ks, altitude_grid, alongtrack_grid, acrosstrack_grid, ecef_to_local = calc_jacobian(df, columns, rows)
    with open(args.jacobian, "wb") as file:
        pickle.dump((y, ks, altitude_grid, alongtrack_grid, acrosstrack_grid, ecef_to_local), file)
    return (y, ks, altitude_grid, alongtrack_grid, acrosstrack_grid, ecef_to_local)


def invert(args, jacobian):
    (y, ks, altitude_grid_edges, alongtrack_grid_edges, acrosstrack_grid_edges, ecef_to_local) = jacobian
    y = y.flatten()
    altitude_grid, alongtrack_grid, acrosstrack_grid = [center_grid(arr) for arr in
        [altitude_grid_edges, alongtrack_grid_edges, acrosstrack_grid_edges]]
    alts = get_local_alts(altitude_grid, acrosstrack_grid, alongtrack_grid, ecef_to_local)
    xa = args.apr_peak_val * mats_inv.generate_xa_from_gaussian(alts, width=args.apr_peak_width,
                                                                meanheight=args.apr_peak_alt)
    Sa_inv, terms = mats_inv.Sa_inv_tikhonov((altitude_grid, acrosstrack_grid * EARTH_RADIUS,
                                              alongtrack_grid * EARTH_RADIUS), args.Sa_weights[0],
                                             args.Sa_weights[1:], store_terms=args.reg_analysis)
    Sa_inv *= (1 / np.max(y)) * 1e6
    Se_inv = sp.diags(np.ones([ks.shape[0]]), 0).astype('float32') * (1 / np.max(y))

    print("Starting inversion...")
    x_hat = mats_inv.do_inversion(ks, y, Sa_inv=Sa_inv, Se_inv=Se_inv, xa=xa, method=args.solver)

    x_hat_reshape1 = np.array(x_hat).reshape(len(altitude_grid), len(acrosstrack_grid), len(alongtrack_grid))

    if args.reg_analysis:
        residual = x_hat - xa.T
        contribs = [residual.T @ term @ residual for term in terms]
        print("Regularisation contributions to misfit:")
        print(f"Zero-order:            {contribs[0]:.1e}")
        print(f"Altitude gradient:     {contribs[1]:.1e}")
        print(f"Across-track gradient: {contribs[2]:.1e}")
        print(f"Along-track gradient:  {contribs[3]:.1e}")

        with open("xa.pkl", "wb") as file:
            pickle.dump((xa.reshape(len(altitude_grid), len(acrosstrack_grid), len(alongtrack_grid)),
                         altitude_grid, alongtrack_grid, acrosstrack_grid, ecef_to_local), file)

    with open(args.out, "wb") as file:
        pickle.dump((x_hat_reshape1, altitude_grid, alongtrack_grid, acrosstrack_grid, ecef_to_local), file)


def center_grid(grid):
    return (grid[:-1] + grid[1:]) / 2


def get_local_alts(radius_grid, acrosstrack_grid, alongtrack_grid, ecef_to_local):
    rr, acrr, alongg = np.meshgrid(radius_grid, acrosstrack_grid, alongtrack_grid, indexing="ij")
    lxx, lyy, lzz = sph2cart(rr, acrr, alongg)
    glgrid = ecef_to_local.inv().apply(np.dstack((lxx.flatten(), lyy.flatten(), lzz.flatten()))[0, :, :])
    altt = rr - geoid_radius(np.arcsin(glgrid[:, 2].reshape(rr.shape) / rr))
    return altt


def main():
    args = get_args()

    read_data = os.path.isfile(args.data)
    read_jacobian = read_data and os.path.isfile(args.jacobian)

    dftop = pd.read_pickle(args.data) if read_data else get_data(args)
    jacobian = pd.read_pickle(args.jacobian) if read_jacobian else get_jacobian(args, dftop)
    del dftop
    invert(args, jacobian)


if __name__ == "__main__":
    main()
