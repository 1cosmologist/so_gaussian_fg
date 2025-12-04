"""
Validation script for Gaussian foreground simulations.
Checks power spectra and cross-spectra against model predictions.
"""

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import skytools as st
import yaml
import os
from pathlib import Path


# Get HEALPix data path from environment variable
hpx_datapath = os.environ.get('SKYTOOLS_DATA', None)


def load_params(params_path):
    """Load parameters from YAML file."""
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)


def dust_scaling(freq, fg_params):
    """Compute dust frequency scaling factor in K_CMB units."""
    freq_piv_dust = fg_params['freq_piv_dust']
    beta_dust = fg_params['beta_dust']
    T_dust = fg_params['T_dust']
    
    U = st.KCMB_to_MJysr(freq) * st.MJysr_to_Kb(freq)
    # Following Planck convention (Planck 2018 XI) of beta_dust - 2 for MBB in antenna temperature units
    return st.greybody(freq, freq_piv_dust, beta_dust - 2., T_dust) / U


def sync_scaling(freq, fg_params):
    """Compute synchrotron frequency scaling factor in K_CMB units."""
    freq_piv_sync = fg_params['freq_piv_sync']
    beta_sync = fg_params['beta_sync']
    
    U = st.KCMB_to_MJysr(freq) * st.MJysr_to_Kb(freq)
    return st.powerlaw(freq, freq_piv_sync, beta_sync) / U


def fg_cl_model(freq1, freq2, fg_params, lmax):
    """
    Compute the model foreground power spectrum (or cross-spectrum) between two frequencies.
    
    Parameters
    ----------
    freq1 : float
        First frequency in GHz
    freq2 : float
        Second frequency in GHz
    fg_params : dict
        Foreground model parameters
    lmax : int
        Maximum multipole
        
    Returns
    -------
    cl_EE, cl_BB : arrays
        Model EE and BB power spectra
    """
    ells = np.arange(lmax + 1)
    Dell_factor = ells * (ells + 1) / (2 * np.pi)
    
    # Dust power spectra at pivot frequency
    A_dust_EE = fg_params['A_dust_EE']
    A_dust_BB = fg_params['A_dust_BB']
    alpha_dust_EE = fg_params['alpha_dust_EE']
    alpha_dust_BB = fg_params['alpha_dust_BB']
    
    cl_dust_EE = np.zeros(ells.shape)
    cl_dust_BB = np.zeros(ells.shape)
    cl_dust_EE[2:] = A_dust_EE * (ells[2:] / 80.) ** alpha_dust_EE
    cl_dust_BB[2:] = A_dust_BB * (ells[2:] / 80.) ** alpha_dust_BB
    cl_dust_EE[2:] /= Dell_factor[2:]
    cl_dust_BB[2:] /= Dell_factor[2:]
    
    # Synchrotron power spectra at pivot frequency
    A_sync_EE = fg_params['A_sync_EE']
    A_sync_BB = fg_params['A_sync_BB']
    alpha_sync_EE = fg_params['alpha_sync_EE']
    alpha_sync_BB = fg_params['alpha_sync_BB']
    
    cl_sync_EE = np.zeros(ells.shape)
    cl_sync_BB = np.zeros(ells.shape)
    cl_sync_EE[2:] = A_sync_EE * (ells[2:] / 80.) ** alpha_sync_EE
    cl_sync_BB[2:] = A_sync_BB * (ells[2:] / 80.) ** alpha_sync_BB
    cl_sync_EE[2:] /= Dell_factor[2:]
    cl_sync_BB[2:] /= Dell_factor[2:]
    
    # Frequency scaling factors
    dust_scale_1 = dust_scaling(freq1, fg_params)
    dust_scale_2 = dust_scaling(freq2, fg_params)
    sync_scale_1 = sync_scaling(freq1, fg_params)
    sync_scale_2 = sync_scaling(freq2, fg_params)
    
    # Cross-spectrum: scale by product of scaling factors
    cl_EE = (dust_scale_1 * dust_scale_2) * cl_dust_EE + (sync_scale_1 * sync_scale_2) * cl_sync_EE
    cl_BB = (dust_scale_1 * dust_scale_2) * cl_dust_BB + (sync_scale_1 * sync_scale_2) * cl_sync_BB
    
    return cl_EE, cl_BB


def validate_fg_simulations(fg_params_path, instr_params_path, output_dir, mc_idx=0):
    """
    Validate foreground simulations against model predictions.
    
    Parameters
    ----------
    fg_params_path : str
        Path to foreground parameters YAML file
    instr_params_path : str
        Path to instrument parameters YAML file
    output_dir : str
        Directory containing the simulation outputs
    mc_idx : int
        Monte Carlo realization index to validate
    """
    # Load parameters
    fg_params = load_params(fg_params_path)
    instr_params = load_params(instr_params_path)
    
    # Get channel info
    channels = list(instr_params.keys())
    freqs = {ch: instr_params[ch]['central_freq_GHz'] for ch in channels}
    beams = {ch: instr_params[ch]['beam_fwhm_arcmin'] for ch in channels}
    nsides = {ch: instr_params[ch]['nside'] for ch in channels}
    
    # Find lowest, highest, and 93 GHz channels
    freq_list = [(ch, freqs[ch]) for ch in channels]
    freq_list_sorted = sorted(freq_list, key=lambda x: x[1])
    
    lowest_ch, lowest_freq = freq_list_sorted[0]
    highest_ch, highest_freq = freq_list_sorted[-1]
    
    # Find 93 GHz channel (MF093)
    mid_ch = None
    mid_freq = None
    for ch, freq in freq_list:
        if 90 <= freq <= 95:
            mid_ch = ch
            mid_freq = freq
            break
    
    if mid_ch is None:
        raise ValueError("Could not find 93 GHz channel")
    
    print("=" * 70)
    print("FOREGROUND SIMULATION VALIDATION")
    print("=" * 70)
    print(f"Validating MC realization: {mc_idx}")
    print(f"Lowest frequency channel: {lowest_ch} ({lowest_freq} GHz)")
    print(f"Middle frequency channel: {mid_ch} ({mid_freq} GHz)")
    print(f"Highest frequency channel: {highest_ch} ({highest_freq} GHz)")
    print("=" * 70)
    
    # Load maps
    output_path = Path(output_dir)
    
    def load_map(ch):
        nside = nsides[ch]
        filename = f"sobs_gaussfg_{ch}_mc{mc_idx:03d}_nside{nside:04d}.fits"
        filepath = output_path / filename
        return hp.read_map(filepath, field=(0, 1, 2)) # type: ignore
    
    map_low = load_map(lowest_ch)
    map_mid = load_map(mid_ch)
    map_high = load_map(highest_ch)
    
    # Compute power spectra
    nside = nsides[lowest_ch]  # Assuming all same nside
    lmax = 1000
    
    print(f"\nComputing power spectra up to lmax = {lmax}...")
    
    # Auto-spectra
    cl_low = hp.anafast(map_low, lmax=lmax, pol=True, nspec=3, use_weights=True, datapath=hpx_datapath)
    cl_mid = hp.anafast(map_mid, lmax=lmax, pol=True, nspec=3, use_weights=True, datapath=hpx_datapath)
    cl_high = hp.anafast(map_high, lmax=lmax, pol=True, nspec=3, use_weights=True, datapath=hpx_datapath)
    
    # Cross-spectra
    cl_low_mid = hp.anafast(map_low, map_mid, lmax=lmax, pol=True, nspec=3, use_weights=True, datapath=hpx_datapath)
    cl_high_mid = hp.anafast(map_high, map_mid, lmax=lmax, pol=True, nspec=3, use_weights=True, datapath=hpx_datapath)
    
    # Compute beam window functions
    Bl_low = hp.gauss_beam(np.deg2rad(beams[lowest_ch] / 60.), lmax=lmax, pol=True)
    Bl_mid = hp.gauss_beam(np.deg2rad(beams[mid_ch] / 60.), lmax=lmax, pol=True)
    Bl_high = hp.gauss_beam(np.deg2rad(beams[highest_ch] / 60.), lmax=lmax, pol=True)
    
    # Debeam the spectra (avoid division by zero)
    ells = np.arange(lmax + 1)
    
    def safe_debeam(cl, Bl1, Bl2, pol_idx):
        """Safely debeam power spectrum."""
        result = np.zeros_like(cl)
        beam_product = Bl1[:, pol_idx] * Bl2[:, pol_idx]
        mask = beam_product > 1e-10
        result[mask] = cl[mask] / beam_product[mask]
        return result
    
    # Debeamed auto-spectra (EE is index 1, BB is index 2)
    cl_low_EE_debeam = safe_debeam(cl_low[1], Bl_low, Bl_low, 1)
    cl_low_BB_debeam = safe_debeam(cl_low[2], Bl_low, Bl_low, 2) # type: ignore
    cl_mid_EE_debeam = safe_debeam(cl_mid[1], Bl_mid, Bl_mid, 1)
    cl_mid_BB_debeam = safe_debeam(cl_mid[2], Bl_mid, Bl_mid, 2) # type: ignore
    cl_high_EE_debeam = safe_debeam(cl_high[1], Bl_high, Bl_high, 1)
    cl_high_BB_debeam = safe_debeam(cl_high[2], Bl_high, Bl_high, 2) # type: ignore
    
    # Debeamed cross-spectra
    cl_low_mid_EE_debeam = safe_debeam(cl_low_mid[1], Bl_low, Bl_mid, 1)
    cl_low_mid_BB_debeam = safe_debeam(cl_low_mid[2], Bl_low, Bl_mid, 2) # pyright: ignore[reportGeneralTypeIssues]
    cl_high_mid_EE_debeam = safe_debeam(cl_high_mid[1], Bl_high, Bl_mid, 1)
    cl_high_mid_BB_debeam = safe_debeam(cl_high_mid[2], Bl_high, Bl_mid, 2)
    
    # Compute model predictions
    print("Computing model predictions...")
    
    cl_model_low_EE, cl_model_low_BB = fg_cl_model(lowest_freq, lowest_freq, fg_params, lmax)
    cl_model_mid_EE, cl_model_mid_BB = fg_cl_model(mid_freq, mid_freq, fg_params, lmax)
    cl_model_high_EE, cl_model_high_BB = fg_cl_model(highest_freq, highest_freq, fg_params, lmax)
    
    cl_model_low_mid_EE, cl_model_low_mid_BB = fg_cl_model(lowest_freq, mid_freq, fg_params, lmax)
    cl_model_high_mid_EE, cl_model_high_mid_BB = fg_cl_model(highest_freq, mid_freq, fg_params, lmax)
    
    # ==================== PLOTTING ====================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Lowest frequency auto-spectrum
    ax = axes[0, 0]
    ax.plot(ells[2:], cl_low_EE_debeam[2:], 'o', ms=2, alpha=0.5, c='C0', label='Sim EE')
    ax.plot(ells[2:], cl_low_BB_debeam[2:], 'o', ms=2, alpha=0.5, c='C1', label='Sim BB')
    ax.plot(ells[2:], cl_model_low_EE[2:], '-', c='C0', label='Model EE')
    ax.plot(ells[2:], cl_model_low_BB[2:], '-', c='C1', label='Model BB')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$C_\ell$ [$\mu K_{CMB}^2$]')
    ax.set_title(f'Auto-spectrum: {lowest_ch} ({lowest_freq} GHz)')
    ax.legend(frameon=False)
    ax.axvline(80, ls=':', c='gray', alpha=0.5)
    
    # Plot 2: Highest frequency auto-spectrum
    ax = axes[0, 1]
    ax.plot(ells[2:], cl_high_EE_debeam[2:], 'o', ms=2, alpha=0.5, c='C0', label='Sim EE')
    ax.plot(ells[2:], cl_high_BB_debeam[2:], 'o', ms=2, alpha=0.5, c='C1', label='Sim BB')
    ax.plot(ells[2:], cl_model_high_EE[2:], '-', c='C0', label='Model EE')
    ax.plot(ells[2:], cl_model_high_BB[2:], '-', c='C1', label='Model BB')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$C_\ell$ [$\mu K_{CMB}^2$]')
    ax.set_title(f'Auto-spectrum: {highest_ch} ({highest_freq} GHz)')
    ax.legend(frameon=False)
    ax.axvline(80, ls=':', c='gray', alpha=0.5)
    
    # Plot 3: Middle frequency auto-spectrum
    ax = axes[0, 2]
    ax.plot(ells[2:], cl_mid_EE_debeam[2:], 'o', ms=2, alpha=0.5, c='C0', label='Sim EE')
    ax.plot(ells[2:], cl_mid_BB_debeam[2:], 'o', ms=2, alpha=0.5, c='C1', label='Sim BB')
    ax.plot(ells[2:], cl_model_mid_EE[2:], '-', c='C0', label='Model EE')
    ax.plot(ells[2:], cl_model_mid_BB[2:], '-', c='C1', label='Model BB')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$C_\ell$ [$\mu K_{CMB}^2$]')
    ax.set_title(f'Auto-spectrum: {mid_ch} ({mid_freq} GHz)')
    ax.legend(frameon=False)
    ax.axvline(80, ls=':', c='gray', alpha=0.5)
    
    # Plot 4: Cross-spectrum lowest x 93 GHz
    ax = axes[1, 0]
    ax.plot(ells[2:], cl_low_mid_EE_debeam[2:], 'o', ms=2, alpha=0.5, c='C0', label='Sim EE')
    ax.plot(ells[2:], cl_low_mid_BB_debeam[2:], 'o', ms=2, alpha=0.5, c='C1', label='Sim BB')
    ax.plot(ells[2:], cl_model_low_mid_EE[2:], '-', c='C0', label='Model EE')
    ax.plot(ells[2:], cl_model_low_mid_BB[2:], '-', c='C1', label='Model BB')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$C_\ell$ [$\mu K_{CMB}^2$]')
    ax.set_title(f'Cross-spectrum: {lowest_ch} x {mid_ch}')
    ax.legend(frameon=False)
    ax.axvline(80, ls=':', c='gray', alpha=0.5)
    
    # Plot 5: Cross-spectrum highest x 93 GHz
    ax = axes[1, 1]
    ax.plot(ells[2:], cl_high_mid_EE_debeam[2:], 'o', ms=2, alpha=0.5, c='C0', label='Sim EE')
    ax.plot(ells[2:], cl_high_mid_BB_debeam[2:], 'o', ms=2, alpha=0.5, c='C1', label='Sim BB')
    ax.plot(ells[2:], cl_model_high_mid_EE[2:], '-', c='C0', label='Model EE')
    ax.plot(ells[2:], cl_model_high_mid_BB[2:], '-', c='C1', label='Model BB')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$C_\ell$ [$\mu K_{CMB}^2$]')
    ax.set_title(f'Cross-spectrum: {highest_ch} x {mid_ch}')
    ax.legend(frameon=False)
    ax.axvline(80, ls=':', c='gray', alpha=0.5)
    
    # Plot 6: Frequency scaling at ell=60 and ell=100
    ax = axes[1, 2]
    
    # Get all frequencies and compute power at ell=60 and ell=100
    all_freqs = sorted([freqs[ch] for ch in channels])
    all_channels_sorted = [ch for ch, _ in freq_list_sorted]
    
    ell_checks = [60, 100]
    power_EE_sim = {ell: [] for ell in ell_checks}
    power_BB_sim = {ell: [] for ell in ell_checks}
    power_EE_model = {ell: [] for ell in ell_checks}
    power_BB_model = {ell: [] for ell in ell_checks}
    
    for ch in all_channels_sorted:
        freq = freqs[ch]
        nside_ch = nsides[ch]
        
        # Load and compute power spectrum
        map_ch = load_map(ch)
        cl_ch = hp.anafast(map_ch, lmax=lmax, pol=True, nspec=3, use_weights=True, datapath=hpx_datapath)
        Bl_ch = hp.gauss_beam(np.deg2rad(beams[ch] / 60.), lmax=lmax, pol=True)
        
        # Model prediction
        cl_model_EE, cl_model_BB = fg_cl_model(freq, freq, fg_params, lmax)
        
        for ell_check in ell_checks:
            # Debeam at ell_check
            cl_EE_debeam = cl_ch[1][ell_check] / (Bl_ch[ell_check, 1] ** 2)
            cl_BB_debeam = cl_ch[2][ell_check] / (Bl_ch[ell_check, 2] ** 2)
            
            power_EE_sim[ell_check].append(cl_EE_debeam)
            power_BB_sim[ell_check].append(cl_BB_debeam)
            power_EE_model[ell_check].append(cl_model_EE[ell_check])
            power_BB_model[ell_check].append(cl_model_BB[ell_check])
    
    # Plot for both ell values
    ax.plot(all_freqs, power_EE_sim[60], 'o', ms=8, c='C0', label=r'Sim EE $\ell=60$')
    ax.plot(all_freqs, power_BB_sim[60], 's', ms=8, c='C1', label=r'Sim BB $\ell=60$')
    ax.plot(all_freqs, power_EE_sim[100], 'o', ms=6, c='C0', mfc='none', label=r'Sim EE $\ell=100$')
    ax.plot(all_freqs, power_BB_sim[100], 's', ms=6, c='C1', mfc='none', label=r'Sim BB $\ell=100$')
    ax.plot(all_freqs, power_EE_model[60], '-', c='C0', label=r'Model EE $\ell=60$')
    ax.plot(all_freqs, power_BB_model[60], '-', c='C1', label=r'Model BB $\ell=60$')
    ax.plot(all_freqs, power_EE_model[100], '--', c='C0', label=r'Model EE $\ell=100$')
    ax.plot(all_freqs, power_BB_model[100], '--', c='C1', label=r'Model BB $\ell=100$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel(r'$C_{\ell}$ [$\mu K_{CMB}^2$]')
    ax.set_title(r'Frequency scaling at $\ell = 60$ and $\ell = 100$')
    ax.legend(frameon=False, fontsize=7, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path / f'validation_mc{mc_idx:03d}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ==================== NUMERICAL CHECKS ====================
    for ell_check in ell_checks:
        print("\n" + "=" * 70)
        print(f"NUMERICAL VALIDATION AT ell = {ell_check}")
        print("=" * 70)
        
        print(f"\n{'Channel':<10} {'Freq [GHz]':<12} {'Sim EE':<15} {'Model EE':<15} {'Ratio':<10}")
        print("-" * 62)
        for i, ch in enumerate(all_channels_sorted):
            freq = freqs[ch]
            ratio = power_EE_sim[ell_check][i] / power_EE_model[ell_check][i]
            print(f"{ch:<10} {freq:<12.1f} {power_EE_sim[ell_check][i]:<15.4e} {power_EE_model[ell_check][i]:<15.4e} {ratio:<10.4f}")
        
        print(f"\n{'Channel':<10} {'Freq [GHz]':<12} {'Sim BB':<15} {'Model BB':<15} {'Ratio':<10}")
        print("-" * 62)
        for i, ch in enumerate(all_channels_sorted):
            freq = freqs[ch]
            ratio = power_BB_sim[ell_check][i] / power_BB_model[ell_check][i]
            print(f"{ch:<10} {freq:<12.1f} {power_BB_sim[ell_check][i]:<15.4e} {power_BB_model[ell_check][i]:<15.4e} {ratio:<10.4f}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return fig


if __name__ == "__main__":
    # Configuration
    FG_PARAMS = "../resources/config/fg_params.yaml"
    INSTR_PARAMS = "../resources/config/instr_params_baseline_pessimistic.yaml"
    OUTPUT_DIR = "../output/foreground_sims/gaussian_fg/"
    MC_IDX = 0
    
    validate_fg_simulations(FG_PARAMS, INSTR_PARAMS, OUTPUT_DIR, MC_IDX)
