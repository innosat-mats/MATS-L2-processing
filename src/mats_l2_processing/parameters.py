from types import SimpleNamespace


def make_conf(conf_type, conf_file, args):
    const, req = {}, {}

    # Variables groups used in multiple configurations
    CCD_VARS = ['channel', 'NCSKIP', 'NRBIN', 'NCOL', 'NRSKIP', 'NROW', 'NCBINCCDColumns', "TEXPMS"]
    ATT_VARS = ["afsAttitudeState", "qprime", 'afsGnssStateJ2000', "EXPDate"]
    TP_VARS = ["TPlat", "TPlon", 'afsTangentPointECEF']
    GEN_RET_VARS = ['START_TIME', 'STOP_TIME', 'CHANNELS', "SA_WEIGHTS", "RET_QTY", "AUX_QTY", "EPSILON_WEIGHTS",
                    "TP_ALT_RANGE", "SCALES", "BOUNDS", "RAD_SCALE", "INTERPOLATOR", "RET_ALT_RANGE", "CG_ATOL",
                    "CG_RTOL", "CG_MAX_STEPS", "TOP_ALT", "STEP_SIZE", "COL_RANGE", "ROW_RANGE", "RT_DATA_FILE",
                    "GRIDDED_PRE", "NCDF_OBS_FACTOR"]
    LM_VARS = ["LM_IT_MAX", "LM_PAR_0", "LM_FAC", "LM_MAX_FACTS_PER_ITER", "CONV_CRITERION"]
    APR_1D_VARS = ["SA_WEIGHTS_1D_APR", "CHANNEL_1D_APR", "MEDCOLS_1D_APR", "GRIDDED_PRE_1D"]

    # NCDF parameters: (<long name>,  <unit>)
    ncpar = {"IR1": ("Infrared image channel 1", "ph/cm^2/s/srad"),
             "IR2": ("Infrared image channel 2", "ph/cm^2/s/srad"),
             "IR3": ("Infrared image channel 3", "ph/cm^2/s/srad"),
             "IR4": ("Infrared image channel 4", "ph/cm^2/s/srad"),
             "UV1": ("Ultraviolet image channel 1", "ph/cm^2/s/srad"),
             "UV2": ("Ultraviolet image channel 2", "ph/cm^2/s/srad"),
             "VER": ("Volume emission rate", "ph/cm^3/s"),
             "T": ("Temperature", "K")}

    # Configuration for temperature iterative solver
    const["iter_T"] = {"NEEDED_DATA": CCD_VARS + ATT_VARS + TP_VARS, "TP_VARS": CCD_VARS + ATT_VARS, "ncpar": ncpar,
                       "POINTING_DATA": ATT_VARS + CCD_VARS}
    req["iter_T"] = GEN_RET_VARS + LM_VARS + ['ALT_GRID', 'ALONG_GRID', 'ACROSS_GRID', "ASPECT_RATIO",
                                              "DISTORTION_CORRECTION", "DISTORTION_DATA", "GEOLOCATE_1D_FROM_TP"] + \
        APR_1D_VARS

    # Configuration for linear
    const["linear_1D"] = const["iter_T"].copy()
    const["linear_1D"]["NEEDED_DATA"] += ["TPECEFx", "TPECEFy", "TPECEFz"]
    req["linear_1D"] = GEN_RET_VARS + ['ALT_GRID', "DISTORTION_CORRECTION", "DISTORTION_DATA", "GEOLOCATE_1D_FROM_TP"]

    # Configuration for IR common grid
    const["superpose"] = {"CCD_VARS": CCD_VARS + ATT_VARS,
                          "ALL_VARS": CCD_VARS + ATT_VARS + ["ImageCalibrated", "ImageDestrayed"],
                          "NCDF_VARS": CCD_VARS + ATT_VARS + TP_VARS + ["CalibrationErrors", "BadColumns"],
                          "POINTING_DATA": ATT_VARS + CCD_VARS,
                          "ncpar": ncpar,
                          "CH_WIDTHS": [3.577769605779391, 8.1656558203897, 3.192647612468147, 3.2126844284028753],
                          "CH_RAYLEIGH_SCALES": [1.10046208, 1.09399409, 1.19713362, 1.0]}
    req["superpose"] = ["RECAL_FAC_IR", "START_TIME", "STOP_TIME", "VERSION", "DISTORTION_CORRECTION", "DISTORTION_DATA"]

    # Configuration for L2 input data preparation
    req["get_data"] = ['START_TIME', 'STOP_TIME', 'VERSION', 'CHANNEL', 'STR_LEN']

    # Configuration for code that generates start/end times for each tomography run in a batch
    req["intervals"] = ['BATCH_START_TIME', 'BATCH_STOP_TIME', 'TOMO_CONDITIONS', 'TOMO_DEFAULT_DURATION',
                        'TOMO_MIN_DURATION', 'TOMO_MIN_OVERLAP']

    # Configuration for data post-processing
    req["post"] = ["GRIDDED_POST", "GRID_TYPE"]

    # Configuration for TP height calculation
    req["heights"] = ["DISTORTION_CORRECTION", "DISTORTION_DATA"]
    const["heights"] = {"POINTING_DATA": ATT_VARS + CCD_VARS, "ncpar": ncpar}
    # const["heights"]["POINTING_DATA"].remove("CCDSEL")
    # const["heights"]["POINTING_DATA"].remove("EXPDate")

    # Default values for all of the above
    defaults = {"SA_WEIGHTS_1D_APR": [1e-1, 1e3],
                "LM_IT_MAX": 9,
                "LM_PAR_0": 0.1,
                "LM_FAC": 10,
                "LM_MAX_FACTS_PER_ITER": 7,
                "EPSILON_WEIGHTS": [1.0, 1.0],
                "TP_ALT_RANGE": (60, 110),
                "RAD_SCALE": 1e14,
                "CONV_CRITERION": 0.96,
                "RET_ALT_RANGE": (60, 108),
                "ASPECT_RATIO": (1, 40, 40),
                "CG_ATOL": 0,
                "CG_RTOL": 1e-5,
                "CG_MAX_STEPS": 5000,
                "VERSION": 0.6,
                "STR_LEN": 20,
                "TOP_ALT": 120e3,
                "STEP_SIZE": 8e3,
                "RECAL_FAC_IR": [1.0, 1.0, 1.0, 1.0],
                "COL_RANGE": [0, 44],
                "ROW_RANGE": [-1, -1],
                'TOMO_DEFAULT_DURATION': 600,
                'TOMO_MIN_DURATION': 480,
                'TOMO_MIN_OVERLAP': 90,
                "CHANNEL_1D_APR": "IR2c",
                "MEDCOLS_1D_APR": 5,
                "INTERPOLATOR": "LINEAR",
                "NCDF_OBS_FACTOR": 1e13,
                "GEOLOCATE_1D_FROM_TP": True}

    if conf_file is not None:
        exec(open(conf_file).read())
    if (conf_type not in req.keys()) and (conf_type not in const.keys()):
        raise ValueError(f"Unknown configuration type {conf_type}!")
    constants = const[conf_type] if conf_type in const.keys() else {}
    # if len(args.keys()) > 0:
    try:
        vargs = vars(args)
    except Exception:
        vargs = {}
    pars = (vargs, dict(globals(), **locals()), defaults)
    # else:
    #    pars = (dict(globals(), **locals()), defaults)
    conf = set_vars(req[conf_type], pars) if conf_type in req.keys() else {}
    return [SimpleNamespace(**dic) for dic in [conf, constants]]


def set_vars(vrs, sources):
    conf = {}
    for var in vrs:
        unset = True
        for src in sources:
            if (var in src) and (src[var] is not None):
                conf[var] = src[var]
                unset = False
                break
        if unset:
            raise ValueError(f"The variable {var} is not set!")
    return conf


def get_updated_conf(conf, new_vars):
    conf_dict = conf.__dict__.copy()
    conf_dict.update(new_vars)
    return SimpleNamespace(**conf_dict)
