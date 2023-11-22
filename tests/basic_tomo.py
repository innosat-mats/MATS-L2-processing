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
# KM_GRID = [[(0, 70, 2), (70, 80, 1), (80, 105, 0.5), (105, 120, 1), (120, 200, 2)],
#            [(-700, -300, 50), (-300, -240, 30), (-240, 240, 20), (240, 300, 30), (300, 700, 50)],
#            200]
KM_GRID = {"IR": [[(0, 58, 2), (58, 60, 1), (62, 107, 0.75), (107, 112, 1), (112, 120, 2)],
                  [(-240, 240, 20)], [(-2500, 2500, 40)]],
           "NLC": [[(0, 58, 1), (58, 90, 0.75), (90, 95, 1), (95, 200, 2)],
                   [(-240, 240, 20)], [(-2500, 2500, 30)]],
           "NLCn": [[(0, 58, 2), (58, 90, 0.75), (90, 92, 1), (92, 200, 3)],
                    [(-180, 180, 10)], [(-2500, -600, 150), (-600, 300, 20), (300, 2500, 150)]],
           "NLCd": [[(0, 75, 3), (78, 80, 1), (80, 90, 0.25), (90, 93, 1), (93, 200, 3)],
                    [(-180, 180, 10)], [(-2500, -600, 150), (-600, 320, 20), (300, 2500, 150)]]
          }


def get_args():
    parser = argparse.ArgumentParser(description="Test 2D tomography",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=str, default="verdec2d.pickle",
                        help="Pickle filename for L1b data. Fetch data from aws and pickle if not found.")
    parser.add_argument("--channel", type=str, default="IR2", help="Channel to retrieve.")
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
    parser.add_argument("--bottom_row", type=int, default=0, help="Number of lowermost row to use.")
    parser.add_argument("--grid", type=str, choices=KM_GRID.keys(), default="IR", help="Choose grid template.")
    parser.add_argument("--processes", type=int, default=4,
                        help="Number of images to process in parallel for jacobian calculation.")
    parser.add_argument("--apr_profile", type=str, help="Sets apriori from the given file (altitude profile pickle)." +
                        " Overrides all other a priori parameters.")
    parser.add_argument("--y", type=str, help="Load measurement data from a specified file, separate from Jacobian.")
    parser.add_argument("--apr_peak_val", type=float, default=3.2e12, help="A priori VER value at peak")
    parser.add_argument("--apr_peak_alt", type=float, default=93000, help="A priori VER peak altitude.")
    parser.add_argument("--apr_peak_width", type=float, default=3700, help="A priori VER peak width.")
    parser.add_argument("--Sa_weights", type=float, nargs=5, default=[1, 500, 20000, 20000, 5e5],
                        help="Tikhonov regularisation weights: [<0-order>, <1-order diff., alt>," +
                        "<1-order diff., across>, <1-order diff., along>, <laplacian>]")
    parser.add_argument("--offset", type=int, default=300, help="Profile selection offset")
    parser.add_argument("--reg_analysis", action="store_true",
                        help="Store xa and calculate additional regularisation parameters")

    return parser.parse_args()


def get_data(args):
    dftop = read_MATS_data(DT.datetime(*args.start_time), DT.datetime(*args.stop_time), level="1b", version="0.4")
    breakpoint()
    with open(args.data, 'wb') as handle:
        pickle.dump(dftop, handle)
    return dftop


def make_grid_proto(proto, offset=0.0, scaling=1.0):
    if (type(proto) is int) or (type(proto) is float):
        return int(proto)
    assert type(proto) is list, "Malformed grid specification!"
    for i, interval in enumerate(proto):
        assert type(interval) is tuple, "Malformed grid specification!"
        assert len(interval) == 3, "Malformed grid specification!"
        if i == 0:
            res = np.arange(*interval)
        else:
            res = np.concatenate((res, np.arange(*interval)))
    return res * scaling + offset


def get_jacobian(args, dftop):
    df = dftop[dftop['channel'] == args.channel].dropna().reset_index(drop=True)
    df = df.loc[args.offset:args.offset + args.nprofiles - 1]
    df = df.reset_index(drop=True)
    columns = np.arange(0, df["NCOL"][0], 1)
    rows = np.arange(args.bottom_row, df["NROW"][0] - 10, 1)
    print(f"Using {len(columns)} columns and {len(rows)} rows.")
    grid_proto = [make_grid_proto(KM_GRID[args.grid][0], scaling=1e3,
                                  offset=geoid_radius(np.deg2rad(np.mean(df['TPlat'])))),
                  make_grid_proto(KM_GRID[args.grid][1], scaling=1e3 / EARTH_RADIUS),
                  make_grid_proto(KM_GRID[args.grid][2], scaling=1e3 / EARTH_RADIUS)]
    print(f"Data set has {len(df)} images from channel {args.channel}. Calculating Jacobian for images {args.offset}-{args.offset + args.nprofiles - 1}")
    jac = calc_jacobian(df, columns, rows, grid_proto=grid_proto, processes=args.processes)
    y, ks, altitude_grid, alongtrack_grid, acrosstrack_grid, ecef_to_local = jac[:6]
    with open(args.jacobian, "wb") as file:
        pickle.dump((y, ks, altitude_grid, alongtrack_grid, acrosstrack_grid, ecef_to_local), file)
    if len(jac) > 6:
        with open(f"tanalts_{args.jacobian}", "wb") as file:
            pickle.dump(jac[6], file)

    return (y, ks, altitude_grid, alongtrack_grid, acrosstrack_grid, ecef_to_local)


def invert(args, jacobian, ye=None):
    (y, ks, altitude_grid_edges, alongtrack_grid_edges, acrosstrack_grid_edges, ecef_to_local) = jacobian
    if ye is not None:  # y is normally calculated with jacobian, but can optionally be provided separately
        assert ye.shape == y.shape
        assert not np.isnan(ye).any()
        y = ye
    y = y.flatten()
    altitude_grid, alongtrack_grid, acrosstrack_grid = [center_grid(arr) for arr in
        [altitude_grid_edges, alongtrack_grid_edges, acrosstrack_grid_edges]]
    alts = get_local_alts(altitude_grid, acrosstrack_grid, alongtrack_grid, ecef_to_local)
    if args.apr_profile:
        with open(args.apr_profile, 'rb') as file:
            profile_xa, profile_alts, _ = pickle.load(file)
        xa = mats_inv.generate_xa_from_alt_profile(alts, profile_xa, profile_alts)
    else:
        xa = args.apr_peak_val * mats_inv.generate_xa_from_gaussian(alts, width=args.apr_peak_width,
                                                                    meanheight=args.apr_peak_alt)
    Sa_inv, terms = mats_inv.Sa_inv_tikhonov((altitude_grid, acrosstrack_grid * EARTH_RADIUS,
                                              alongtrack_grid * EARTH_RADIUS), args.Sa_weights[0],
                                             diff_weights=args.Sa_weights[1:4], laplacian_weight=args.Sa_weights[4],
                                             store_terms=args.reg_analysis, volume_factors=True)
    Sa_inv *= (1 / np.max(y)) * 1e6
    Se_inv = sp.diags(np.ones([ks.shape[0]]), 0).astype('float32') * (1 / np.max(y))

    print("Starting inversion...")
    x_hat = mats_inv.do_inversion(ks, y, Sa_inv=Sa_inv, Se_inv=Se_inv, xa=xa, method=args.solver)

    x_hat_reshape1 = np.array(x_hat).reshape(len(altitude_grid), len(acrosstrack_grid), len(alongtrack_grid))

    if args.reg_analysis:
        residual = x_hat - xa.T
        print("Regularisation contributions to misfit:")
        with open(f'reg_info_{args.out}.txt', 'a') as file:
            for name, matrix in terms.items():
                info = f"{name}: {residual.T @ matrix @ residual:.1e}"
                print(info)
                file.write(info)

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
    jacobian = jacobian[:6]
    y = pd.read_pickle(args.y) if args.y else None
    invert(args, jacobian, ye=y)


if __name__ == "__main__":
    main()
