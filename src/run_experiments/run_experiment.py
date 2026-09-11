import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../../')
from data_assim.assimilation_nanny import AssimilationConductor



# --- Specify truth
dir_truth = Path("/home/maxf/projects/REFORM/pgradDA/grasp/ecmwf_ensemble")
OBS_FILE = Path(f"{dir_truth}/synthetic_observation.nc")
ds_obs_true = xr.open_dataset(OBS_FILE)


# --- Specify experiemnt details

## CONFIG
config_path = Path("/home/maxf/projects/REFORM/pgradDA/src/run_experiments")

## Experiment dir
base_dir = Path("/home/maxf/projects/REFORM/pgradDA/grasp/case_study/run_single") #/2026/02/10/00
sim_ctrl_dir = base_dir / "2026/02/10/00"



# ===================================
# START ASSIMILATION
# ===================================
conductor = AssimilationConductor(
    config_path=config_path,
    simdir_ctrl=sim_ctrl_dir,
    base_dir=base_dir,
    ztop_assim_idx= 30, #30,
    obs_file=OBS_FILE,
    save_weights=False,
    gpu=True,   
)
conductor()