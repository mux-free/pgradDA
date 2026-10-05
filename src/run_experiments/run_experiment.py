import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import sys
sys.path.insert(1, '../../')
from data_assim.assimilation_nanny import AssimilationConductor
from simdata_processing.save_ensemble import save_summary_ds
from data_assim.create_synthetic_obs import synthobs_from_truth



# ============================================================================================
# Load Config
# -----------
def _load_config(ymlpath:Path):
    """ Load config.yml file """
    with ymlpath.open("r") as file:
        return yaml.safe_load(file)
config_path = Path("/home/maxf/projects/REFORM/pgradDA/src/run_experiments")
config = _load_config(config_path / "config.yml")
r_std = config["ASSIMILATION"]["OBSERVATIONS"]["obs_std"]
# ============================================================================================


# ============================================================================================
# Directories
# ------------
truth_dir = Path("/home/maxf/projects/REFORM/pgradDA/grasp/case_study/run_truth/")
obs_file = truth_dir / "synthetic_obs/synthetic_observation.nc"
base_dir = Path("/home/maxf/projects/REFORM/pgradDA/grasp/case_study")
# ============================================================================================


# ----------------------------------------------------------------
# -- EXPERIMENT DIR
exp_name = "run_control"
exp_name = "run_exp3_dense"
exp_dir = base_dir / exp_name
# ----------------------------------------------------------------


# ============================================================================================
# -- Create synthetic observations
create_synthetic_observations = True

xcoord = []
ycoord = []
for i in np.arange(16,128,32):
    for j in np.arange(16,128,32):
        xcoord.append(int(i))
        ycoord.append(int(j))

if create_synthetic_observations:
    truth_simdata_path = truth_dir / "run/2026/02/10/00"
    synthobs_from_truth(
        truth_simdata_path = truth_simdata_path,
        file_synth_obs = obs_file ,
        sigma_obs = r_std,
        obs_ix = xcoord, #[32, 96, 64, 32, 96]
        obs_iy = ycoord, #[32, 32, 64, 96, 96]
        ztop_obs = 2000,
        z_step = 5,
        ti_start = 2,
        t_step = 3,
        )
# ============================================================================================


# ============================================================================================
# START ASSIMILATION
# ===================================
conductor = AssimilationConductor(
    config_path=config_path,
    base_dir=exp_dir,
    ztop_assim_idx= 30,
    obs_file=obs_file,
    save_weights=False,
    gpu=True,
)
conductor(do_assimilation=True, run_control=False)

# ============================================================================================

save_summary_ds(exp_dir, obs_file=obs_file, spectra_analysis=False, spectra_enspert=False)
print(f"FINISHED EVERYTHING IN: {exp_dir}")