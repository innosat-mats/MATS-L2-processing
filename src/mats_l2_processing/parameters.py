from types import SimpleNamespace


def make_conf(conf_type, conf_file, args):
    const, req = {}, {}

    # Variables groups used in multiple configurations
    CCD_VARS = ['CCDSEL', 'NCSKIP', 'NRBIN', 'NCOL', 'NRSKIP', 'NROW', 'NCBINCCDColumns', "TEXPMS"]
    ATT_VARS = ["qprime", "afsAttitudeState", 'afsGnssStateJ2000', "EXPDate"]
    TP_VARS = ["TPlat", "TPlon", 'afsTangentPointECI']

    # Configuration for temperature iterative solver
    const["iter_T"] = {"NEEDED_DATA": CCD_VARS + ATT_VARS + TP_VARS, "TP_VARS": CCD_VARS + ATT_VARS}
    req["iter_T"] = ['START_TIME', 'STOP_TIME', 'ALT_GRID', 'ALONG_GRID', 'ACROSS_GRID', 'CHANNELS', "LM_IT_MAX",
                     "LM_PAR_0", "LM_FAC", "LM_MAX_FACTS_PER_ITER", "SA_WEIGHTS", "RET_QTY", "AUX_QTY",
                     "EPSILON_WEIGHTS", "TP_ALT_RANGE", "SCALES", "BOUNDS", "RAD_SCALE", "INTERPOLATOR",
                     "CONV_CRITERION", "RET_ALT_RANGE", "ASPECT_RATIO", "CG_ATOL", "CG_RTOL", "CG_MAX_STEPS",
                     "TOP_ALT", "STEP_SIZE", "COL_RANGE", "SA_WEIGHTS_1D_APR", "CHANNEL_1D_APR", "MEDCOLS_1D_APR"]

    # Configuration for IR common grid
    const["superpose"] = {"CCD_VARS": CCD_VARS + ATT_VARS,
                          "ALL_VARS": CCD_VARS + ATT_VARS + ["ImageCalibrated"],
                          "NCDF_VARS": CCD_VARS + ATT_VARS + TP_VARS + ["CalibrationErrors", "BadColumns"]}
    req["superpose"] = ["RECAL_FAC_IR", "START_TIME", "STOP_TIME", "VERSION"]

    # Configuration for L2 input data preparation
    req["get_data"] = ['START_TIME', 'STOP_TIME', 'VERSION', 'CHANNEL', 'STR_LEN']

    # Configuration for code that generates start/end times for each tomography run in a batch
    req["intervals"] = ['BATCH_START_TIME', 'BATCH_STOP_TIME', 'TOMO_CONDITIONS', 'TOMO_DEFAULT_DURATION',
                        'TOMO_MIN_DURATION', 'TOMO_MIN_OVERLAP']

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
                'TOMO_DEFAULT_DURATION': 600,
                'TOMO_MIN_DURATION': 480,
                'TOMO_MIN_OVERLAP': 90,
                "CHANNEL_1D_APR": "IR2c",
                "MEDCOLS_1D_APR": 5,
                "INTERPOLATOR": "LINEAR"}

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
