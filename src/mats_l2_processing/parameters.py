from types import SimpleNamespace


def make_conf(conf_type, conf_file, args):
    const, req = {}, {}
    const["iter_T"] = {"NEEDED_DATA": ["NCOL", "NROW", "TPlat", "TPlon", "EXPDate", "qprime", 'afsAttitudeState',
                                       'afsGnssStateJ2000', 'CCDSEL', 'NCSKIP', 'NCBINCCDColumns', 'NRSKIP', 'NRBIN',
                                       "IR1c", "IR2c", 'afsTangentPointECI', 'TEXPMS']}
    req["iter_T"] = ['START_TIME', 'STOP_TIME', 'ALT_GRID', 'ALONG_GRID', 'ACROSS_GRID', 'CHANNEL', "LM_IT_MAX",
                     "LM_PAR_0", "LM_FAC", "LM_MAX_FACTS_PER_ITER", "SA_WEIGHTS_T", "SA_WEIGHTS_VER", "EPSILON_IR1",
                     "EPSILON_IR2", "TP_ALT_RANGE", "VER_SCALE", "T_SCALE", "VER_BOUNDS", "T_BOUNDS", "RAD_SCALE",
                     "CONV_CRITERION", "RET_ALT_RANGE", "ASPECT_RATIO", "CG_ATOL", "CG_RTOL", "CG_MAX_STEPS",
                     "TOP_ALT", "STEP_SIZE"]

    req["get_data"] = ['START_TIME', 'STOP_TIME', 'VERSION', 'CHANNEL', 'STR_LEN']

    defaults = {"SA_WEIGHTS_T": [1, 500, 20000, 20000, 5e5],
                "SA_WEIGHTS_VER": [1, 500, 20000, 20000, 5e5],
                "LM_IT_MAX": 9,
                "LM_PAR_0": 0.1,
                "LM_FAC": 10,
                "LM_MAX_FACTS_PER_ITER": 7,
                "EPSILON_IR1": 1.0,
                "EPSILON_IR2": 1.0,
                "TP_ALT_RANGE": (60, 110),
                "VER_SCALE": 1e4,
                "T_SCALE": 1e2,
                "VER_BOUNDS": (0, 2e8),
                "T_BOUNDS": (100, 600),
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
                "STEP_SIZE": 8e3}

    if conf_file is not None:
        exec(open(conf_file).read())
    if conf_type not in req.keys():
        raise ValueError(f"Unknown configuration type {conf_type}!")
    constants = const[conf_type] if conf_type in const.keys() else {}
    conf = set_vars(req[conf_type], (vars(args), dict(globals(), **locals()), defaults))
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
