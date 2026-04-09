from types import SimpleNamespace


def make_conf(conf_type, conf_file, args):
    const, req = {}, {}

    # Variables groups used in multiple configurations
    CCD_VARS = ['channel', 'NCSKIP', 'NRBIN', 'NCOL', 'NRSKIP', 'NROW', 'NCBINCCDColumns', "TEXPMS"]
    ATT_VARS = ["afsAttitudeState", "qprime", 'afsGnssStateJ2000', "time"]
    TP_VARS = ["TPlat", "TPlon", "TPECEFx", "TPECEFy", "TPECEFz"]
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
             "UV1": ("Ultraviolet image channel 1", "ph/cm^2/nm/s/srad"),
             "UV2": ("Ultraviolet image channel 2", "ph/cm^2/nm/s/srad"),
             "VER": ("Volume emission rate", "ph/cm^3/s"),
             "dVER": ("Directional spectral emission rate", "ph/cm^3/nm/s/srad"),
             "T": ("Temperature", "K")}

    # Various approximate constants for simple estimations needed in the code
    SAT_SPEED_APPROX = 7.55  # km/s

    # -------------------------------------------------------------------------------------------------------------

    # Configuration for temperature iterative solver
    const["iter_T"] = {"NEEDED_DATA": CCD_VARS + ATT_VARS + TP_VARS, "TP_VARS": CCD_VARS + ATT_VARS, "ncpar": ncpar,
                       "POINTING_DATA": ATT_VARS + CCD_VARS, "sat_speed_approx": SAT_SPEED_APPROX}
    req["iter_T"] = ['ALT_GRID', 'ALONG_GRID', 'ACROSS_GRID', "ASPECT_RATIO", "SEP_CHN_LOS", "OBS_SRC_VAR",
                     "DISTORTION_CORRECTION", "DISTORTION_DATA", "GEOLOCATE_1D_FROM_TP"] + \
        GEN_RET_VARS + LM_VARS + APR_1D_VARS

    # Configuration for linear
    const["linear_1D"] = const["iter_T"].copy()
    const["linear_1D"]["NEEDED_DATA"] += ["TPECEFx", "TPECEFy", "TPECEFz"]
    req["linear_1D"] = GEN_RET_VARS + ['ALT_GRID', "DISTORTION_CORRECTION", "DISTORTION_DATA", "GEOLOCATE_1D_FROM_TP"]

    # Configuration for stray light removal
    req["destray"] = ["SCAT_MAX_SZA", "FIT_REF_ROWS", "FIT_BOT_ROW", "IRB_DENOISE_HW", "IRB_DENOISE_THR",
                      "ZEMAX_DATA_DIR", "TRANSMISSIVITY"]
    const["destray"] = {"IRB_Y_PITCH": 0.01778459, "IRB_R_SCALE_HEIGHT": 0.1524675}

    # Configuration for IR common grid
    const["superpose"] = {"CCD_VARS": CCD_VARS + ATT_VARS + ["TPsza"],
                          "ALL_VARS": CCD_VARS + ATT_VARS + ["ImageCalibrated"],
                          "IRB_VARS": ["ImageStrayScattered", "ImageDescat"],
                          "NCDF_VARS": CCD_VARS + ATT_VARS + TP_VARS + ["CalibrationErrors", "BadColumns"],
                          "POINTING_DATA": ATT_VARS + CCD_VARS,
                          "ncpar": ncpar,
                          "CH_WIDTHS": [3.577769605779391, 8.1656558203897, 3.192647612468147, 3.2126844284028753],
                          "CHN_WIDTHS": {"IR1": 3.577769605779391, "IR2": 8.1656558203897,
                                         "IR3": 3.192647612468147, "IR4": 3.2126844284028753},
                          "CH_RAYLEIGH_SCALES": [1.10046208, 1.09399409, 1.19713362, 1.0],
                          "CHN_RAYLEIGH_SCALES": {"IR1": 1.10046208, "IR2": 1.09399409, "IR3": 1.19713362, "IR4": 1.0}}
    const["superpose"].update(const["destray"])
    req["superpose"] = ["RECAL_FAC_IR", "DISTORTION_CORRECTION", "DISTORTION_DATA", "SCAT_MAX_SZA", "ZEMAX_DATA_DIR",
                        "TRANSMISSIVITY", "WRITE_IRB_CONTRIBUTION", "SCAT_TRANSFER", "SEP_IRB_DESTRAY"]

    # Configuration for L2 input data preparation
    req["get_data"] = ['START_TIME', 'STOP_TIME', 'VERSION', 'CHANNEL', 'STR_LEN']

    # Configuration for code that generates start/end times for each tomography run in a batch
    req["intervals"] = ['BATCH_START_TIME', 'BATCH_STOP_TIME', 'TOMO_CONDITIONS', 'TOMO_DEFAULT_DURATION',
                        'TOMO_MIN_DURATION', 'TOMO_MIN_OVERLAP']

    # Configuration for data post-processing
    req["post"] = ["GRIDDED_POST", "GRID_TYPE"]

    # Dummy conf
    req["deg_map"] = ["CHANNELS", "DISTORTION_CORRECTION", "SEP_CHN_LOS", "DISTORTION_DATA", 'DEG_WRT_SAT_AXIS']
    const["deg_map"] = {"POINTING_DATA": ATT_VARS + CCD_VARS}

    # Configuration for TP height calculation
    req["heights"] = ["DISTORTION_CORRECTION", "DISTORTION_DATA", "PLANET_FILE"]
    const["heights"] = {"POINTING_DATA": ATT_VARS + CCD_VARS, "ncpar": ncpar}
    # const["heights"]["POINTING_DATA"].remove("CCDSEL")
    # const["heights"]["POINTING_DATA"].remove("time")

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
                'TOMO_DEFAULT_DURATION': 660,
                'TOMO_MIN_DURATION': 500,
                'TOMO_MIN_OVERLAP': 50,
                "CHANNEL_1D_APR": "IR2c",
                "MEDCOLS_1D_APR": 5,
                "INTERPOLATOR": "LINEAR",
                "NCDF_OBS_FACTOR": 1.0,
                "GEOLOCATE_1D_FROM_TP": True,
                "SEP_CHN_LOS": False,
                "SCAT_MAX_SZA": 90,
                "FIT_REF_ROWS": {"IR3": [(45, 50)] + [(53, 58)] * 8, "IR4": [(49, 54)] + [(53, 58)] * 8},
                "FIT_BOT_ROW": {"IR3": 0, "IR4": 0},
                "IRB_DENOISE_HW": [1, 2, 2],
                "IRB_DENOISE_THR": 3,
                "WRITE_IRB_CONTRIBUTION": False,
                "TRANSMISSIVITY": {"IR1": 1.0, "IR2": 1.0},
                "SCAT_TRANSFER": False,
                "SEP_IRB_DESTRAY": False,
                }

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
