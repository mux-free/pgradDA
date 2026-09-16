from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd



@dataclass(frozen=True, slots=True)  # Note slots=True, helps to make datalcass faster int erms of attribute acess
class ExperimentContext:
    """
    Immutable container for all constants of the EnsGen methods 
    that both the EnsembleConductor and every MemberNanny need.
    """


    # ----------------------------
    # -- Set all the attributes
    # ----------------------------
    config: dict = field(repr=False)
    n_members: int = field(init=False)
    # --- Timing
    dt_spinup: int     = field(init=False)
    t0_spinup:datetime = field(init=False)
    t0_da: datetime    = field(init=False)
    n_da: int          = field(init=False)
    dt_da: int       = field(init=False)
    dt_pred: int     = field(init=False)    
    t0_pred: datetime  = field(init=False)
    t_end: datetime    = field(init=False)
    
    # Paths
    experiment_dir: Path = field(init=False)
    simdir_ctrl: Path = field(init=False)

    # --- DA specifics
    radius: int       = field(init=False)
    vert_loc:int|None = field(init=False)
    inflation: float  = field(init=False)
    mode_4d_3d: str   = field(init=False)
    r2: float         = field(init=False)
    rtpp: float|bool  = field(init=False)
    state_vars: list  = field(init=False)


    # ----------------------------
    # -- Actual initisation --> As data-class is frozen, use __setattr__
    # ----------------------------
    def __post_init__(self):
        config = self.config

        # --- 2) Run specification
        n_members = config["RUN"]["n_members"]
    
        # --- 3) TIMING
        dt_spinup = pd.Timedelta(config["RUN"]["dt_spinup"]).total_seconds()
        n_da      = config["ASSIMILATION"]["n_da"]
        dt_da     = pd.Timedelta(config["ASSIMILATION"]["dt_da"]).total_seconds()
        dt_pred   = pd.Timedelta(config["ASSIMILATION"]["dt_pred"]).total_seconds()
        
        t0_spinup = datetime.strptime(config["TIMING"]["t0_spinup"], "%Y/%m/%d/%H")
        t0_da     = t0_spinup + timedelta(seconds=dt_spinup)
        t0_pred   = t0_da + timedelta(seconds=(n_da-1)*dt_da)
        t_end     = t0_pred + timedelta(seconds=dt_pred)
        
        # --- 4) DA specifications
        radius        = config["ASSIMILATION"]["LETKF"]["loc_radius"]
        vert_loc      = config["ASSIMILATION"]["LETKF"]["loc_vertical"]
        inflation     = config["ASSIMILATION"]["LETKF"]["inflation"]
        mode_4d_3d    = config["ASSIMILATION"]["LETKF"]["mode_4d_3d"]
        state_vars    = config["ASSIMILATION"]["LETKF"]["state_vars"]
        rtpp          = config["ASSIMILATION"]["LETKF"]["rtpp"]
        r2            = config["ASSIMILATION"]["OBSERVATIONS"]["R2"]

        # --- 5) Handle paths
        experiment_dir = config["PATHS"]["experiment_dir"]
        simdir_ctrl = experiment_dir / "run" / t0_spinup.strftime("%Y/%m/%d/%H")

    
        # --- 6)  __init__             Set fields as attribtues (must be done in this way, as we specify contex to be immmutable dataclass)
        object.__setattr__(self, "n_members",  n_members)
        object.__setattr__(self, "n_da",       n_da)
        object.__setattr__(self, "dt_da",      dt_da)
        object.__setattr__(self, "dt_pred", dt_pred)
        object.__setattr__(self, "dt_spinup",  dt_spinup)
        object.__setattr__(self, "t0_spinup",         t0_spinup)
        object.__setattr__(self, "t0_da", t0_da)
        object.__setattr__(self, "t0_pred",    t0_pred)
        object.__setattr__(self, "t_end",    t_end)
        object.__setattr__(self, "experiment_dir", experiment_dir)
        object.__setattr__(self, "simdir_ctrl", simdir_ctrl)
        # -- DA specifics
        object.__setattr__(self, "radius",     radius)
        object.__setattr__(self, "vert_loc",   vert_loc)
        object.__setattr__(self, "inflation",  inflation)
        object.__setattr__(self, "mode_4d_3d", mode_4d_3d)
        object.__setattr__(self, "state_vars", state_vars)
        object.__setattr__(self, "rtpp",       rtpp)
        object.__setattr__(self, "r2",         r2)
        object.__setattr__(self, "experiment_dir",  experiment_dir)
        object.__setattr__(self, "simdir_ctrl",  simdir_ctrl)




