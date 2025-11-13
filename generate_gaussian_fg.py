"""
Script to generate Gaussian foreground simulations for SO channels.
Based on the foreground model from Wolz et al. 2024.
"""

import healpy as hp
import numpy as np
import skytools as st
import os
from pathlib import Path
from tqdm import tqdm


class GaussianForegroundSimulator:
    """
    Class to simulate polarized foreground maps (dust + synchrotron)
    for Simons Observatory frequency channels.
    """
    
    def __init__(self):
        # Dust model parameters
        self.A_dust_EE = 56  # uK_CMB^2
        self.A_dust_BB = 28  # uK_CMB^2
        self.alpha_dust_EE = -0.32
        self.alpha_dust_BB = -0.16
        self.freq_piv_dust = 353.  # GHz
        self.beta_dust = 1.54
        self.T_dust = 20.  # K
        
        # Synchrotron model parameters
        self.A_sync_EE = 9.  # uK_CMB^2
        self.A_sync_BB = 1.6  # uK_CMB^2
        self.alpha_sync_EE = -0.7
        self.alpha_sync_BB = -0.93
        self.freq_piv_sync = 23.  # GHz
        self.beta_sync = -3.
        
        # SO instrument parameters
        self.so_freqs = np.array([21, 39, 93, 145, 225, 280])  # GHz
        self.so_beams = np.array([91., 63., 30., 17., 11., 9.])  # arcmin
        self.so_nside = np.array([128, 128, 512, 512, 1024, 1024])
        self.lmax = 3 * int(self.so_nside.max()) - 1
        
        # Precompute power spectra
        self.cl_dust_EE, self.cl_dust_BB = self.dust_cl()
        self.cl_sync_EE, self.cl_sync_BB = self.sync_cl()
    
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
        freq_iter = zip(self.so_freqs, self.so_beams, self.so_nside)
        for i, (freq, beam, nside) in enumerate(tqdm(freq_iter, total=len(self.so_freqs), 
                                                       desc=f"  Sim {sim_idx:04d}", 
                                                       leave=False, ncols=120)):
            fg_Q, fg_U = self.generate_fg_map(freq, beam, nside, 
                                              alms_d_EE, alms_d_BB, 
                                              alms_s_EE, alms_s_BB)
            
            # Create output filename
            filename = f"sobs_gaussfg_freq{freq:03d}GHz_mc{sim_idx:03d}_nside{nside:04d}.fits"
            filepath = Path(output_dir) / filename
            
            # Save as FITS file with Q and U in separate columns
            # Create T=0, Q, U map
            fg_map = np.array([np.zeros(fg_Q.shape), fg_Q, fg_U])
            
            # Create header with metadata
            header = [
                ('UNITS', 'uK_CMB', 'Map units'),
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
        print(f"SO frequencies: {self.so_freqs} GHz")
        print(f"SO nsides: {self.so_nside}")
        print("-" * 60)
        
        for sim_idx in tqdm(range(n_sims), desc="Generating simulations", ncols=120):
            self.generate_simulation(sim_idx, output_path)
        
        print("-" * 60)
        print(f"Completed {n_sims} simulations!")
        print(f"Total maps generated: {n_sims * len(self.so_freqs)}")


def main():
    """Main function to run 100 simulations."""
    # Initialize simulator
    simulator = GaussianForegroundSimulator()
    
    # Set output directory
    output_dir = "../output/foreground_sims"
    
    # Run 100 simulations
    simulator.run_simulations(n_sims=5, output_dir=output_dir)


if __name__ == "__main__":
    main()
