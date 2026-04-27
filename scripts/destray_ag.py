import argparse
import numpy as np

from mats_l2_processing.obs_preprocessing import find_images, cross_maps, reinterpolate_irreg, deghost
from mats_l2_processing.io import read_L1_ncdf, add_ncdf_vars, ncdf_filter_dim
from mats_l2_processing.parameters import make_conf, get_updated_conf
from mats_l2_processing.pointing import Pointing
# from mats_l2_processing.util import seconds2DT


def get_args():
    parser = argparse.ArgumentParser(description="Stray light removal for IR1 and IR2",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ag_file", type=str, help="Netcdf file with L1b data for IR1 or IR2")
    parser.add_argument("--conf", type=str, default="conf.py", help="Read configuration from file.")
    parser.add_argument("--irb_files", type=str, nargs=2, help="Netcdf file with L1b data for IR3 and IR4.")
    parser.add_argument("--out_file", type=str, help="Output file name. Overrides --overwrite.")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite input with output. Invalid image data will be lost!")
    return parser.parse_args()


def main():
    args = get_args()
    conf, const = make_conf("superpose", args.conf, args)

    # Read in data
    agd = read_L1_ncdf(args.ag_file, var=const.CCD_VARS + ["ImageCalibrated"], center_times=True)
    chns = [agd["channel"][0], "IR3", "IR4"]
    conf = get_updated_conf(conf, {"CHANNELS": chns, "SEP_CHN_LOS": True})
    if chns[0] not in ["IR1", "IR2"]:
        raise ValueError("This script is for stray light removal in IR1 and IR2, but got {ag_chn} data instead!")
    recal = {chn: conf.RECAL_FAC_IR[i] for i, chn in enumerate(["IR1", "IR2", "IR3", "IR4"])}

    ird = {chns[0]: agd}
    for i, chn in enumerate(chns[1:]):
        if args.irb_files is None:
            if chns[0] in args.ag_file:
                irf = args.ag_file.replace(chns[0], chn)
            else:
                raise ValueError(f"{chn} file neither provided nor could be guessed from {chns[0]} file name.")
        else:
            irf = args.irb_files[i]
        data = read_L1_ncdf(irf, var=const.CCD_VARS + ["ImageStrayScattered", "ImageDescat"], center_times=True)
        assert data["channel"][0] == chn, f"Got {data['channel'][0]} file instead of {chn} file!"
        ird[chn] = data

    # Load info aboput channel properties
    info = {chn: np.load(f"{conf.ZEMAX_DATA_DIR}/zemax_info_{chn}.npz") for chn in chns}
    if conf.SCAT_TRANSFER:
        scat_own = {chn: np.load(f"{conf.ZEMAX_DATA_DIR}/{chn}_scat_on_{chn}_grid.npz")['scattered']
                    for chn in chns[1:]}
        scat_transfer = {chn: np.load(f"{conf.ZEMAX_DATA_DIR}/{chns[0]}_scat_on_{chn}_grid.npz")['scattered']
                         for chn in chns[1:]}
    else:
        scat_own = {chn: np.ones(ird[chn]["ImageDescat"].shape[2]) for chn in chns[1:]}
        scat_transfer = scat_own

    # Filter daylight images, identify with corresponding background images
    valid = agd["TPsza"] < conf.SCAT_MAX_SZA
    idx = {}
    valid, idx["IR3"], idx["IR4"] = find_images(ird[chns[0]]["time_s"], valid,
                                                [ird[chn]["time_s"] for chn in chns[1:]])

    # Relative pointing of the channels
    pointing = Pointing([ird[chn] for chn in chns], conf, const)
    deg_maps = [np.swapaxes(pointing.chn_map(chn=chn), 0, 2) for chn in chns]
    cmaps = cross_maps([ird[chn] for chn in chns], const.CCD_VARS, deg_maps)

    # Background channel contributions
    rayleigh_factor = {chn: const.CHN_RAYLEIGH_SCALES[chn] *
                       (1 + info[chn]["ghost_strength"] * np.exp(info[chn]["ghost_offset"] / const.IRB_R_SCALE_HEIGHT))
                       for chn in chns}
    print(rayleigh_factor)
    scat_factor = {chn: scat_transfer[chn] / scat_own[chn] * conf.TRANSMISSIVITY[chns[0]] for chn in chns[1:]}
    contrib = {chn: recal[chn] * (ird[chn]["ImageStrayScattered"] * scat_factor[chn][np.newaxis, np.newaxis, :] +
               ird[chn]["ImageDescat"] * rayleigh_factor[chns[0]] / rayleigh_factor[chn]) for chn in chns[1:]}

    # Reinterpolation to the airglow channel
    r_contrib = {chn: reinterpolate_irreg(contrib[chn], idx[chn], deg_maps[i + 1], cmaps[i + 1]) 
                 for i, chn in enumerate(chns[1:])}

    # Remove stray light
    if conf.SEP_IRB_DESTRAY:
        irb_ch = {"IR1": "IR4", "IR2": "IR3"}
        destrayed = ird[chns[0]]["ImageCalibrated"][valid, :, :] * recal[chns[0]] - r_contrib[irb_ch[chns[0]]]
    else:
        destrayed = ird[chns[0]]["ImageCalibrated"][valid, :, :] * recal[chns[0]] - 0.5 * (r_contrib["IR3"] + r_contrib["IR4"])

    non_deghosted = destrayed.copy()  # For debugging.
    destrayed = deghost(destrayed, info[chns[0]]["ghost_strength"], offset_angle=info[chns[0]]["ghost_offset"],
                        deg_map=deg_maps[0], pad=0.0)
    destrayed *= const.CHN_WIDTHS[chns[0]]  # Convert spectral radiance to radience

    # Write output
    if args.out_file is None:
        if args.overwrite:
            out_file = args.ag_file
        elif "L1b" in args.ag_file:
            out_file = args.ag_file.replace("L1b", "L1c_b")
        elif args.ag_file.endswith(".nc"):
            out_file = args.ag_file.replace(".nc", "_destrayed.nc")
        else:
            out_file = f"{args.ag_file}_destrayed.nc"
    else:
        out_file = args.ofile

    ncdf_filter_dim(args.ag_file, "time", np.where(valid)[0], out_file)  # Copy of the input file with invalid images removed
    outvar = [("ImageFinal", "Image to be used in Level 2 processing", np.maximum(destrayed, 0.0)),
              ("ImageBeforeDeghost", "Non-deghosted image (for testing)", non_deghosted)]
    if conf.WRITE_IRB_CONTRIBUTION:
        outvar += [(f"{chn}Contribution", f"Contribution of {chn} in stray light removal", r_contrib[chn])
                   for chn in chns[1:]]
    add_ncdf_vars(out_file, "ImageCalibrated", outvar, units=[("ImageFinal", "photon meter-2 steradian-1 second-1")])

if __name__ == "__main__":
    main()
