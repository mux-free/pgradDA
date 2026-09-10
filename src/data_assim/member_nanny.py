import shutil
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta

from abc import abstractmethod

# Add .../REFORM/src to sys.path, s.t. i can improt my own modelus!!
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# OWN MODULES
import sys
sys.path.insert(1, '../../')
from run_experiments.experiment_context import ExperimentContext



class MemberNanny:
    """
    Performs operations on ONE ensemble member. Context is loaded that provides gloabal (immutable) attributes.
    Responsibility of class:
        - Initialise ensemble folders (.../data_assim/run/yyyy/mm/dd/hh_<ID>) as copy of reference
        - Run (aspire) spinup for all ensemble members (incl. control)
    """

    def __init__(self, 
                 context: ExperimentContext,
                 simdir_ctrl: Path,
                 member_nr: int | str | None,
                 verbose: int = 0) -> None:


        # --- reference the shared context ───────────────────────────────
        self.context = context
        self.verbose = verbose
        self.member_nr  = member_nr
        self.member_id = self._retireve_ensemble_tag()


        # --- Define inner domain corridor where no perturbations are added

        # --- Ensemble shortcuts
        self.n_members         = context.n_members

        # --- shortcuts for the fields your methods already use 
        self.dt_spinup: int  = context.dt_spinup
        self.t0_da: datetime = context.t0_da        
        self.dt_da: int      = context.dt_da        


        # --- Member-specific stuff (control and member directory)
        self.simdir_ctrl: Path = simdir_ctrl
        self.member_dir: Path  = simdir_ctrl / self.member_id





    def _retireve_ensemble_tag(self):
        """ 
        Return the ensemble ID 
        Logic: ensemble have an integer as member id, 
        while control have string (e.g. 'ctrl')
        """
        if isinstance(self.member_nr, int) and self.member_nr>0:
            tag = f"ensemble_{int(self.member_nr):02d}" 
        elif isinstance(self.member_nr, str) and (self.member_nr.lower()=="ctrl" or self.member_nr.lower()=="ref"):
            tag = "" 
        elif self.member_nr==0:
            tag = ""
        else:
            raise ValueError(f"ERROR 404: Member-ID {self.member_nr} not found")
        return tag



    def _ensure_presence_first_restartfile(self):
        """ Tests if first restart file is present """
        memb_dir_files = [p.name for p in self.member_dir.glob("*")]
        t0_da = self.t0_da.strftime("%y%m%d%H%M")

        expected = f"graspInRestart_{t0_da}.meso.nc"

        if expected not in memb_dir_files:
            raise FileExistsError(
                f"First Restart-File does not exist.\n"
                f"Expected: {expected}\n"
                f"Directory: {self.member_dir}"
            )

    def update_timing(self, tstart_window: datetime, dt_run:float, debug:bool=False):
        """ 
        Updates attributes accodring to current time step of ensemble 
            --> Method is called at every new time-step if it is an iterative scheme (like breeding)
            --> Based on this, restart-files are renamed to make their name time-aware

        NOTE: Important function, test for bugs
        """

        if not isinstance(tstart_window, datetime):
            self.tstart_window = datetime.strptime(tstart_window, "%Y/%m/%d/%H/%M")
        else:
            self.tstart_window = tstart_window
        
        # Length of current cycle    
        dtcycle = int(dt_run)
       
        # --- 2) Get End-Date (start-date + cycle-length)
        self.tend_window            = tstart_window + timedelta(seconds=dtcycle)
        self.tstart_previous_window = tstart_window - timedelta(seconds=dtcycle)

        # --- 3) Get string timetag in format YYYYmmDDHHMM
        self.timetag:str          = tstart_window.strftime('%Y%m%d%H%M')
        self.timetag_next:str     = self.tend_window.strftime('%Y%m%d%H%M')
        self.timetag_previous:str = self.tstart_previous_window.strftime('%Y%m%d%H%M')

        if debug:
            print()
            print(f"Timetag (now): {self.timetag}")
            print(f"timetag (t-1): {self.timetag_previous}")
            print(f"Timetag (t+1): {self.timetag_next}")
            print()




    # ====================================================================================================================



    def rename_output_files(self, restart_present=True, debug=False):
        """ Rename graspOutRestart and graspOutSimdata """
        # Create old fild-names and path (that will be changed)
        member_dir = Path(self.member_dir)
        
        simdata_plane = member_dir / "graspOutSimdata.meso.nc"
        simdata_plane.rename(member_dir / f"graspOutSimdata_{self.timetag_next}.meso.nc")
        
        if restart_present:
            restart_plane = member_dir / "graspOutRestart.meso.nc"
            restart_plane.rename(member_dir / f"graspOutRestart_{self.timetag_next}.meso.nc")

        # For control runs, copy graspOut to graspIn rest
        if restart_present and self.member_id == "":  # Identifier for control run
            shutil.copy(
                member_dir / f"graspOutRestart_{self.timetag_next}.meso.nc",
                member_dir / f"graspInRestart_{self.timetag_next}.meso.nc"
            )

        if debug:
            print(f"Rename file: {member_dir}/graspOutSimdata_{self.timetag_next}.meso.nc")
            if restart_present:
                print(f"Rename file: {member_dir}/graspOutRestart_{self.timetag_next}.meso.nc")


  
    def save(self, ds:xr.Dataset, fname:str, add_timetag:bool=True):
        """ 
        Saves dataset (either perturbation or restart file)

        fname is either "graspOutPerturbations" or "graspInRestart"
        """
        app = f"_{self.timetag}" if add_timetag else ""
        savepath = self.member_dir / f"{fname}{app}.meso.nc" 
        
        utils.save_ds(ds, path=savepath)


