"""
Script to generate Gaussian foreground simulations for SO channels.
Based on the foreground model from Wolz et al. 2024.
"""

import healpy as hp
import numpy as np
import skytools as st
import os
import yaml
from pathlib import Path
from tqdm import tqdm


def load_params(params):
    """
    Load parameters from a dictionary or YAML file path.
    
    Parameters
    ----------
    params : dict or str or Path
        Either a dictionary of parameters or a path to a YAML file
        
    Returns
    -------
    dict
        Dictionary of parameters
    """
    if params is None:
        return {}
    elif isinstance(params, dict):
        return params
    elif isinstance(params, (str, Path)):
        with open(params, 'r') as f:
            loaded = yaml.safe_load(f)
            return loaded if loaded is not None else {}
    else:
        raise ValueError(f"params must be a dict or path to YAML file, got {type(params)}")


class GaussianForegroundSimulator:
    """
    Class to simulate polarized foreground maps (dust + synchrotron)
    for Simons Observatory frequency channels.
    
    Parameters
    ----------
    fg_params : dict or str or Path, optional
        Foreground model parameters. Can be a dictionary or path to a YAML file.
        Keys can include:
        - A_dust_EE, A_dust_BB: Dust amplitudes in uK_CMB^2
        - alpha_dust_EE, alpha_dust_BB: Dust power law indices
        - freq_piv_dust: Dust pivot frequency in GHz
        - beta_dust: Dust spectral index
        - T_dust: Dust temperature in K
        - A_sync_EE, A_sync_BB: Synchrotron amplitudes in uK_CMB^2
        - alpha_sync_EE, alpha_sync_BB: Synchrotron power law indices
        - freq_piv_sync: Synchrotron pivot frequency in GHz
        - beta_sync: Synchrotron spectral index
    instr_params : dict or str or Path, optional
        Instrument parameters. Can be a dictionary or path to a YAML file.
        Keys can include:
        - freqs: List of frequencies in GHz
        - beams: List of beam FWHMs in arcmin
        - nsides: List of HEALPix nside values
    """
    
    def __init__(self, fg_params=None, instr_params=None):
        # Load parameters from dict or YAML file
        fg = load_params(fg_params)
        instr = load_params(instr_params)
        
        # Dust model parameters (with defaults)
        self.A_dust_EE = fg.get('A_dust_EE', 56)  # uK_CMB^2
        self.A_dust_BB = fg.get('A_dust_BB', 28)  # uK_CMB^2
        self.alpha_dust_EE = fg.get('alpha_dust_EE', -0.32)
        self.alpha_dust_BB = fg.get('alpha_dust_BB', -0.16)
        self.freq_piv_dust = fg.get('freq_piv_dust', 353.)  # GHz
        self.beta_dust = fg.get('beta_dust', 1.54)
        self.T_dust = fg.get('T_dust', 20.)  # K
        
        # Synchrotron model parameters (with defaults)
        self.A_sync_EE = fg.get('A_sync_EE', 9.)  # uK_CMB^2
        self.A_sync_BB = fg.get('A_sync_BB', 1.6)  # uK_CMB^2
        self.alpha_sync_EE = fg.get('alpha_sync_EE', -0.7)
        self.alpha_sync_BB = fg.get('alpha_sync_BB', -0.93)
        self.freq_piv_sync = fg.get('freq_piv_sync', 23.)  # GHz
        self.beta_sync = fg.get('beta_sync', -3.)
        
        # Instrument parameters - handle nested format or flat format
        self._parse_instr_params(instr)
        self.lmax = 3 * int(self.instr_nside.max()) - 1
        
        # Precompute power spectra
        self.cl_dust_EE, self.cl_dust_BB = self.dust_cl()
        self.cl_sync_EE, self.cl_sync_BB = self.sync_cl()
    
    def _parse_instr_params(self, instr):
        """
        Parse instrument parameters from either nested or flat format.
        
        Nested format (per-channel):
            LF21:
                central_freq_GHz: 21.
                beam_fwhm_arcmin: 91.
                nside: 512
        
        Flat format:
            freqs: [21, 39, ...]
            beams: [91., 63., ...]
            nsides: [512, 512, ...]
        """
        # Default values
        default_channels = ['LF021', 'LF039', 'MF093', 'MF145', 'HF225', 'HF280']
        default_freqs = [21., 39., 93., 145., 225., 280.]
        default_beams = [91., 63., 30., 17., 11., 9.]
        default_nsides = [512, 512, 512, 512, 512, 512]
        
        if not instr:
            # Use defaults
            self.channel_names = default_channels
            self.instr_freqs = np.array(default_freqs)
            self.instr_beams = np.array(default_beams)
            self.instr_nside = np.array(default_nsides)
            return
        
        # Check if it's nested format (first value is a dict)
        first_value = next(iter(instr.values()), None)
        if isinstance(first_value, dict):
            # Nested format - extract from per-channel dictionaries
            self.channel_names = list(instr.keys())
            self.instr_freqs = np.array([instr[ch].get('central_freq_GHz', instr[ch].get('freq')) 
                                         for ch in self.channel_names])
            self.instr_beams = np.array([instr[ch].get('beam_fwhm_arcmin', instr[ch].get('beam')) 
                                         for ch in self.channel_names])
            self.instr_nside = np.array([instr[ch].get('nside', 512) 
                                         for ch in self.channel_names])
        else:
            # Flat format
            self.channel_names = instr.get('channel_names', default_channels)
            self.instr_freqs = np.array(instr.get('freqs', default_freqs))
            self.instr_beams = np.array(instr.get('beams', default_beams))
            self.instr_nside = np.array(instr.get('nsides', default_nsides))
    
    def dust_cl(self):
        """Compute dust power spectra at pivot frequency."""
        ells = np.arange(self.lmax + 1)
        Dell_factor = ells * (ells + 1) / (2 * np.pi)
        
        cl_dust_EE = np.zeros(ells.shape)
        cl_dust_BB = np.zeros(ells.shape)
        
        cl_dust_EE[2:] = self.A_dust_EE * (ells[2:] / 80.) ** self.alpha_dust_EE
        cl_dust_BB[2:] = self.A_dust_BB * (ells[2:] / 80.) ** self.alpha_dust_BB
        
        cl_dust_EE[2:] /= Dell_factor[2:]
        cl_dust_BB[2:] /= Dell_factor[2:]
        
        return cl_dust_EE, cl_dust_BB
    
    def sync_cl(self):
        """Compute synchrotron power spectra at pivot frequency."""
        ells = np.arange(self.lmax + 1)
        Dell_factor = ells * (ells + 1) / (2 * np.pi)
        
        cl_sync_EE = np.zeros(ells.shape)
        cl_sync_BB = np.zeros(ells.shape)
        
        cl_sync_EE[2:] = self.A_sync_EE * (ells[2:] / 80.) ** self.alpha_sync_EE
        cl_sync_BB[2:] = self.A_sync_BB * (ells[2:] / 80.) ** self.alpha_sync_BB
        
        cl_sync_EE[2:] /= Dell_factor[2:]
        cl_sync_BB[2:] /= Dell_factor[2:]
        
        return cl_sync_EE, cl_sync_BB
    
    def dust_scaling(self, freq):
        """Compute dust frequency scaling factor."""
        U = st.KCMB_to_MJysr(freq) * st.MJysr_to_Kb(freq)
        # Following Planck convention (Planck 2018 XI) of beta_dust - 2 for MBB in antenna temperature units
        return st.greybody(freq, self.freq_piv_dust, self.beta_dust - 2., self.T_dust) / U
    
    def sync_scaling(self, freq):
        """Compute synchrotron frequency scaling factor."""
        U = st.KCMB_to_MJysr(freq) * st.MJysr_to_Kb(freq)
        return st.powerlaw(freq, self.freq_piv_sync, self.beta_sync) / U
    
    def dust_alms_at_pivot(self):
        """Generate dust alms at pivot frequency."""
        alms_d_EE = hp.synalm(self.cl_dust_EE, lmax=self.lmax)
        alms_d_BB = hp.synalm(self.cl_dust_BB, lmax=self.lmax)
        return alms_d_EE, alms_d_BB
    
    def sync_alms_at_pivot(self):
        """Generate synchrotron alms at pivot frequency."""
        alms_s_EE = hp.synalm(self.cl_sync_EE, lmax=self.lmax)
        alms_s_BB = hp.synalm(self.cl_sync_BB, lmax=self.lmax)
        return alms_s_EE, alms_s_BB
    
    def generate_fg_map(self, freq, beam_fwhm, nside, alms_d_EE, alms_d_BB, alms_s_EE, alms_s_BB):
        """
        Generate foreground Q and U maps for a given frequency channel.
        
        Parameters
        ----------
        freq : float
            Frequency in GHz
        beam_fwhm : float
            Gaussian beam FWHM in arcmin
        nside : int
            HEALPix nside parameter
        alms_d_EE, alms_d_BB : array
            Dust alms at pivot frequency
        alms_s_EE, alms_s_BB : array
            Synchrotron alms at pivot frequency
            
        Returns
        -------
        fg_map_Q, fg_map_U : arrays
            Foreground Q and U maps
        """
        dust_scale = self.dust_scaling(freq)
        sync_scale = self.sync_scaling(freq)
        
        alms_fg_EE = dust_scale * alms_d_EE + sync_scale * alms_s_EE
        alms_fg_BB = dust_scale * alms_d_BB + sync_scale * alms_s_BB
        
        lmax_loc = 3 * nside - 1
        bl = hp.gauss_beam(np.deg2rad(beam_fwhm / 60.), lmax=self.lmax, pol=True)[0:lmax_loc+1]
        alms_fg_EE = hp.almxfl(alms_fg_EE, bl[:, 1])
        alms_fg_BB = hp.almxfl(alms_fg_BB, bl[:, 2])
        
        fg_map_Q, fg_map_U = hp.alm2map_spin([alms_fg_EE, alms_fg_BB], nside, 2, lmax=self.lmax)
        
        return fg_map_Q, fg_map_U
    
    def generate_simulation(self, sim_idx, output_dir):
        """
        Generate one complete simulation for all SO frequency channels.
        
        Parameters
        ----------
        sim_idx : int
            Simulation index
        output_dir : str or Path
            Directory to save the maps
        """
        # Set random seed for reproducibility
        np.random.seed(112359 + sim_idx)
        
        # Generate alms at pivot frequencies (one realization per simulation)
        alms_d_EE, alms_d_BB = self.dust_alms_at_pivot()
        alms_s_EE, alms_s_BB = self.sync_alms_at_pivot()
        
        # Generate maps for each frequency channel
        freq_iter = zip(self.channel_names, self.instr_freqs, self.instr_beams, self.instr_nside)
        for i, (ch_name, freq, beam, nside) in enumerate(tqdm(freq_iter, total=len(self.instr_freqs), 
                                                       desc=f"  Sim {sim_idx:04d}", 
                                                       leave=False, ncols=120)):
            fg_Q, fg_U = self.generate_fg_map(freq, beam, nside, 
                                              alms_d_EE, alms_d_BB, 
                                              alms_s_EE, alms_s_BB)
            
            # Create output filename using channel name
            filename = f"sobs_gaussfg_{ch_name}_mc{sim_idx:03d}_nside{nside:04d}.fits"
            filepath = Path(output_dir) / filename
            
            # Save as FITS file with Q and U in separate columns
            # Create T=0, Q, U map
            fg_map = np.array([np.zeros(fg_Q.shape), fg_Q, fg_U])
            
            # Create header with metadata
            header = [
                ('UNITS', 'uK_CMB', 'Map units'),
                ('CHANNEL', ch_name, 'Channel name'),
                ('FREQ', freq, 'Frequency in GHz'),
                ('BEAM', beam, 'Beam FWHM in arcmin'),
                ('SIMIDX', sim_idx, 'Simulation index'),
            ]
            
            hp.write_map(filepath, fg_map, overwrite=True, dtype=np.float32, 
                        extra_header=header)
    
    def run_simulations(self, n_sims, output_dir):
        """
        Run multiple simulations and save to disk.
        
        Parameters
        ----------
        n_sims : int
            Number of simulations to generate
        output_dir : str or Path
            Directory to save the maps
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {n_sims} foreground simulations...")
        print(f"Output directory: {output_path.absolute()}")
        print(f"Channels: {self.channel_names}")
        print(f"Frequencies: {self.instr_freqs} GHz")
        print(f"Nsides: {self.instr_nside}")
        print("-" * 60)
        
        for sim_idx in tqdm(range(n_sims), desc="Generating simulations", ncols=120):
            self.generate_simulation(sim_idx, output_path)
        
        print("-" * 60)
        print(f"Completed {n_sims} simulations!")
        print(f"Total maps generated: {n_sims * len(self.instr_freqs)}")

