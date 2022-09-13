import numpy as np
from VoigtFit.funcs.voigt import Voigt
from VoigtFit.funcs.voigt import convolve_numba
from scipy.signal import fftconvolve, gaussian
from astropy.io import fits
from astropy.table import Table

from py4most import etc
from py4most.etc.qmost_etc import (get_transmission,
                                   get_sky_model,
                                   get_efficiency)


MgII = [('MgII_2796', 'MgII', 2796.3523, 0.629, 2.692e+08, 24.305),
        ('MgII_2803', 'MgII', 2803.531, 0.308, 2.692e+08, 24.305)]

def add_mgII_absorption(wl, flux):
    # from z = 0.4 to z = 2.3
    # logN = 13.4 (corresponds to W_rest = 1 Å for MgII_2796)
    z_grid = [0.5, 1.0, 1.5, 2.0]
    logN_grid = [12.4, 12.9, 13.4, 14.4, 15.4]
    N = len(logN_grid)
    norm_offset = (np.arange(N) - N // 2) / (N // 2)
    # Create slightly offset absorption systems at each redshift:
    abs_offset = 0.1
    z_offset = norm_offset * abs_offset
    # Use fixed b for all:
    b = 15.    # km/s

    # Calculate optical depth:
    tau = np.zeros_like(wl)
    all_redshifts = list()
    for z0 in z_grid:
        for dz, logN in zip(z_offset, logN_grid):
            z = z0 + dz
            all_redshifts.append(z)
            for trans in MgII:
                l0 = trans[2]
                f = trans[3]
                gam = trans[4]
                tau += Voigt(wl, l0, f, 10**logN, b*1.e5, gam, z=z)

    P = np.exp(-tau)

    # Load resolution kernel:
    R = fits.getdata('/Users/krogager/Projects/4MOST/py4most/static/specres_matrix_4most_LR.fits')
    kernel = R.T
    Npad = kernel.shape[1]//2
    pad = np.ones(Npad)
    P_pad = np.concatenate((pad, P, pad))
    P_con = np.zeros_like(P)
    for i, lsf_i in enumerate(kernel):
        P_con[i] = np.sum(P_pad[i:i+2*Npad+1] * lsf_i)
    
    return all_redshifts, P, P_con


if __name__ == '__main__':

    redshift = 2.5
    # S/N ~ 10 per Å
    G = 20.1
    # S/N ~ 1 per Å
    #G = 22.5

    t_exp = 3600

    temp = etc.make_quasar_template(
                redshift, G,
                lya_forest=False,
    )
    temp_wl, temp_flux, DLA_list = temp
    temp_flux = np.interp(etc.qmost_etc.wl_joint, temp_wl, temp_flux)
    temp_wl = etc.qmost_etc.wl_joint

    # Add MgII absorption
    z_grid, P0, P_abs = add_mgII_absorption(temp_wl, temp_flux)
    temp_flux_abs = temp_flux * P_abs

    # Create Mock Observation
    mock_fname = 'qso_z%.1f_G%.1f_obs.fits' % (redshift, G)
    airmass = 'mid'         # airmass = 1.2
    moon_phase = 'dark'     # FLI = 0.2
    transmission = get_transmission(airmass=airmass)
    sky_model = get_sky_model(airmass, moon_phase)
    throughput = get_efficiency(airmass)
    wl, flux, err, qual = etc.apply_noise(
                                      temp_wl, temp_flux_abs, sky_model, throughput, transmission,
                                      t_exp=t_exp,
                                      filename=mock_fname,
                                      cr=False,
                                      )

    # Calculate median SNR
    pixsize = 0.25
    cut = (wl > 6000) & (wl < 6100)
    SNR = flux / err
    SNR_pix = np.nanmedian(SNR[cut])
    npix = 1. / pixsize
    SNR_dl = SNR_pix * np.sqrt(npix)

    # Analyze absorption:
    norm_flux = flux / temp_flux
    norm_err = err / temp_flux
    line_stats = list()
    for z in z_grid:
        for trans in MgII:
            l0 = trans[2]
            lmin = l0*(z+1) - 2.
            lmax = l0*(z+1) + 2.
            window = (wl > lmin) & (wl < lmax)

            # Calculate observed equiv. width:
            W_obs = np.sum(1. - norm_flux[window]) * pixsize
            W_err = np.sqrt(np.sum(norm_err[window]**2)) * pixsize

            # Calculate intrinsic equiv. width:
            W_int = np.sum(1. - P_abs[window]) * pixsize
            line_stats.append([l0*(z+1), W_int, W_obs, W_err])

    tab = Table(rows=line_stats, names=['lambda_obs', 'W_int', 'W_obs', 'W_err'])
    tab.write('lines_SNR-%.0f.csv' % SNR_dl, overwrite=True)

