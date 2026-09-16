import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../../')
from data_assim.assimilation_nanny import AssimilationConductor
from simdata_processing.save_ensemble import save_summary_ds


# --- Specify truth
dir_truth = Path("/home/maxf/projects/REFORM/pgradDA/grasp/ecmwf_ensemble")
OBS_FILE = Path(f"{dir_truth}/synthetic_observation.nc")
ds_obs_true = xr.open_dataset(OBS_FILE)


# --- Specify experiemnt details

## CONFIG
config_path = Path("/home/maxf/projects/REFORM/pgradDA/src/run_experiments")

## Experiment dir
base_dir = Path("/home/maxf/projects/REFORM/pgradDA/grasp/case_study")
# exp_name = "run_ensemble"
exp_name = "run_single"

# ----------------------------------------------------------------
# -- Base direction of experiment
exp_dir = base_dir / exp_name
# ----------------------------------------------------------------


# ===================================
# START ASSIMILATION
# ===================================
conductor = AssimilationConductor(
    config_path=config_path,
    base_dir=exp_dir,
    ztop_assim_idx= 30, #30,
    obs_file=OBS_FILE,
    save_weights=False,
    gpu=True,   
)
conductor(do_assimilation=True)

save_summary_ds(exp_dir, obs_file=OBS_FILE, spectra_analysis=False, spectra_enspert=False)
print(f"FINISHED EVERYTHING IN: {exp_dir}")