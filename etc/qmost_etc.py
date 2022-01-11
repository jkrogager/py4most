#!/usr/local/opt/python@3.8/bin/python3.8
"""
Quasar simulator for 4MOST

The quasar spectra are generated based on the X-shooter template by Selsing et al. (2016)
To see the documentation, run:
    python3 qmost_qso_etc.py --help

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d
from astropy.io import fits
from astropy.table import Table

import os
import sys

from py4most.etc.dust import FM2007, MW_reddening
from py4most.etc import lya

## --airmass = 'high'  # AIRMASS = 1.05 / 1.2 / 1.45
airmass_conversion = {
        'high': 1.45,
        'mid': 1.2,
        'low': 1.05,
        }

## --moon_phase = 'dark' # DARK: FLI=0.2,  GREY: FLI=0.5
sky_conversion = {
        'dark': 0.2,
        'grey': 0.5,
        }

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'data/')

# Gaia G-band filter curve:
G_filter = np.loadtxt(data_path+'Gaia_G.tab')

# Load the Quasar Template from Selsing et al. 2016
quasar_template = fits.getdata(data_path+'quasar_template_Selsing2016.fits')

# Load List of Narrow Lines
nl_linelist = np.loadtxt(data_path+'narrow_lines.txt', usecols=(1, 2, 3))

# Calculate extinction curve in quasar rest-frame
pars_smc = (-4.959, 2.264, 0.389, 0.461, 4.57, 0.94, 2.74)
ksi_dust = FM2007(1.e4/quasar_template['WAVE'], pars_smc)


# Read noise per pixel:
RON = {'blue': 2.286,
       'green': 2.265,
       'red': 2.206,
       }
all_arms = ['blue', 'green', 'red']

# Final Wavelength Grid for joint spectrum:
N_pix_in_spectrum = 23201
wl_joint = np.linspace(3700., 9500, N_pix_in_spectrum)

# Galactic extinction curve:
ksi_mw = MW_reddening(wl_joint, Rv=3.1)

# Cosmic Ray Rate:
# ~1.0e-5 cosmic rays per spectral pixel per minute
CR_rate_per_sec = 1.0e-5/60


def get_transmission(airmass='mid'):
    # Include various airmasses (mid, low, high)...
    sky_trans = fits.getdata(data_path+'sky_transmission.fits')
    trans_joint = np.interp(wl_joint, sky_trans['WAVE'], sky_trans['TRANS'], right=1., left=1.)
    return trans_joint

def get_sky_model(airmass='mid', moon_phase='dark'):
    sky_model_per_arm = dict()
    for arm in all_arms:
        # Sky model for given airmass and moon phase:
        model_data = fits.getdata(data_path+'sky_model_%s.fits' % moon_phase, arm)
        sky_model_per_arm[arm] = {'WAVE': model_data['WAVE'], 'FLUX': model_data[airmass]}
    return sky_model_per_arm


def get_efficiency(airmass='mid'):
    """
    Returns a Dict with three keys: blue, green, red.
    Each entry is a FITS Table with the given total system efficiency per arm:
        WAVE  Q_EFF
    """
    # Instrument, atmosphere, telescope + fibre efficiencies:
    system_eff = dict()
    for arm in all_arms:
        tab_throughput = fits.getdata(data_path+'efficiency_4most.fits', arm)
        system_eff[arm] = tab_throughput[airmass]
    return system_eff


def synthetic_G_band(wavelength, flux):
    T = np.interp(wavelength, G_filter[:, 0], G_filter[:, 1], left=0, right=0)
    synphot = np.sum(flux*T)/np.sum(T)
    return synphot


def make_narrow_lines(z):
    strength = 10**np.random.normal(0.8, 0.3)
    v = np.random.normal(600, 200) / 2.35 * (z+1)
    wl = quasar_template['WAVE'].copy()
    flux_nl = np.zeros_like(wl)
    for l0, fmin, fmax in nl_linelist:
        if np.abs(l0-4960.30) < 0.1:
            f_rel /= 3.
        else:
            f_rel = np.random.uniform(fmin, fmax)
        sig = v/299792*l0
        flux_nl += np.exp(-0.5*(wl-l0)**2/sig**2) / np.sqrt(2*np.pi) / sig * f_rel
    return flux_nl*strength


def make_quasar_template(z, mag, Av=0., Ebv=0., lya_forest=True, absorption=None, filename='', fwhm_vel=None, narrow_lines=True):
    """
    z : quasar redshift
    mag : the Gaia magnitude in G-band (Vega mag), roughly corresponds to SDSS r-band
    Av : extinction in rest-frame V-band (assuming SMC extinction)
    Ebv : Galactic extinction, E(B-V)
    lya_forest : include a synthetic Lyman-alpha forest?
    absorption : an optional array of absorption lines (normalized to 1 in the continuum)
                 must be calculated on the same grid as the quasar template (i.e., transformed
                 to the quasar rest-frame)
    filename : if given, the template is saved as a FITS table
    fwhm_vel : velocity broadening to be applied
    narrow_lines : include variable narrow lines?
    """
    # Add Vega to AB offset (incl. empirical offset to match ESO ETC):
    mag -= 0.03

    # Redshift and Normalize Template:
    temp_wl = quasar_template['WAVE'].copy() * (z+1)
    temp_flux = quasar_template['FLUX'].copy()
    f0 = synthetic_G_band(temp_wl, temp_flux)
    wl_band = 6424.9269
    f_band = 1./(wl_band)**2 * 10**(-(mag+2.406)/2.5)
    temp_flux = temp_flux/f0*f_band

    # Add narrow lines:
    if narrow_lines:
        flux_nl = make_narrow_lines(z)
        temp_flux += flux_nl/f0*f_band

    if fwhm_vel is not None:
        # broaden template:
        kernel = fwhm_vel / 20. / 2.35
        temp_flux = gaussian_filter1d(temp_flux, kernel)

    if Av > 0:
        temp_flux = temp_flux * 10**(-0.4*ksi_dust*Av)

    if Ebv > 0:
        # assume average Rv = 3.1
        Av_gal = Ebv * 3.1
        temp_flux = temp_flux * 10**(-0.4*ksi_mw*Av_gal)

    if absorption is not None:
        temp_flux *= absorption

    DLA_list = []
    if lya_forest:
        if z > 2.1:
            wl_lya, T_lya, DLA_list = lya.lya_transmission(2.0, z, R=4000.)
            lya_profile = np.interp(temp_wl, wl_lya, T_lya, right=1., left=0.)
            temp_flux = temp_flux * lya_profile
            if len(DLA_list) > 0:
                # Add metals -- not used for now:
                pass

    if filename:
        save_template(filename, temp_wl, temp_flux, DLA_list, z, mag, Av, lya_forest)

    return temp_wl, temp_flux, DLA_list



def save_template(filename, wl, flux, DLA_list, z, mag, Av, lya_forest):
    hdu = fits.HDUList()

    hdr = fits.Header()
    hdr['AUTHOR'] = 'Spectral Generator'
    hdr['COMMENT'] = 'Synthetic quasar model spectrum'
    hdr['REDSHIFT'] = z
    hdr['MAG'] = (mag, "Gaia G-band (Vega)")
    hdr['AV'] = Av
    hdr['FOREST'] = (lya_forest, "Incl. Lya Forest?")
    prim = fits.PrimaryHDU(header=hdr)
    hdu.append(prim)

    col_wl = fits.Column(name='WAVE', array=wl, format='D', unit='Angstrom')
    col_flux = fits.Column(name='FLUX', array=flux, format='D', unit='erg/s/cm2/A')
    tab = fits.BinTableHDU.from_columns([col_wl, col_flux])
    tab.name = 'TEMPLATE'
    hdu.append(tab)

    if len(DLA_list) > 0:
        DLAs = np.array(DLA_list)
        z_DLA = DLAs[:, 0]
        NHI_DLA = DLAs[:, 1]
    else:
        z_DLA = np.array([])
        NHI_DLA = np.array([])
    col_z = fits.Column(name='Z_DLA', array=z_DLA, format='E')
    col_logNHI = fits.Column(name='LOG_NHI', array=NHI_DLA, format='E')
    tab2 = fits.BinTableHDU.from_columns([col_z, col_logNHI])
    tab2.name = 'DLAS'
    hdu.append(tab2)

    hdu.writeto(filename, overwrite=True, output_verify='silentfix')


def save_mock_spectrum(filename, wl, flux, err, qual, t_exp):
    N = len(wl)
    hdu = fits.HDUList()

    hdr = fits.Header()
    hdr['EXPTIME']  = (t_exp, 'Total integration time per data element (s)')
    hdr['ORIGIN']   = ('ESO-PARANAL', 'Observatory or facility')
    hdr['TELESCOP'] = ('ESO-VISTA', 'ESO telescope designation')
    hdr['INSTRUME'] = ('QMOST   ', 'Instrument name')

    prim = fits.PrimaryHDU(header=hdr)
    hdu.append(prim)

    # `pre` is the length of the array at 0.25Å sampling
    pre = '23201'
    col_wl = fits.Column(name='WAVE', array=np.array([wl]),
                         format=pre+'E', unit='Angstrom', disp='F9.4')
    col_flux = fits.Column(name='FLUX', array=np.array([flux]),
                           format=pre+'E', unit='erg/(s*cm^2*Angstrom)', disp='E13.5E2')
    col_err = fits.Column(name='ERR_FLUX', array=np.array([err]),
                          format=pre+'E', unit='erg/(s*cm^2*Angstrom)', disp='E13.5E2')
    col_qual = fits.Column(name='QUAL', array=np.array([qual]),
                           format=pre+'J', null=-1, disp='I1')
    col_noss = fits.Column(name='FLUX_NOSS', array=np.array([flux*0.]),
                           format=pre+'E', unit='erg/(s*cm^2*Angstrom)', disp='E13.5E2')
    col_noss_err = fits.Column(name='ERR_FLUX_NOSS', array=np.array([err*0.]),
                               format=pre+'E', unit='erg/(s*cm^2*Angstrom)', disp='E13.5E2')
    tab = fits.BinTableHDU.from_columns([col_wl, col_flux, col_err, col_qual, col_noss, col_noss_err])
    tab.name = 'PHASE3SPECTRUM'
    hdu.append(tab)

    hdu.writeto(filename, overwrite=True, output_verify='silentfix')


def apply_noise(temp_wl, temp_flux, sky_model, throughput, transmission,
                t_exp=1200, filename=''):
    """
    temp_wl : array of wavelengths in the input template (units: Angstrom)
    temp_flux : array of fluxes in the input template (units: erg/s/cm2/A)
    sky_model : FITS-Table with three extensions {BLUE, GREEN, RED}
                The table contains the following columns: WAVE, FLUX
    throughput : FITS-Table of effective throughput including instrument, fibre and telescope
                 with three extensions {BLUE, GREEN, RED}
    transmission : array of atmospheric transmission model (telluric absorption).
                   Other absorption features can be included as well, such as metal absorption lines.
    t_exp : Exposure time in seconds (default=1200)
    filename : Filename of the noisy spectrum. If not given, the spectrum is not saved.
    """
    # Collecting area of 4.1m telescope:
    # A_tel = np.pi*((4.1/2)**2 - (1.2/2)**2)*1.e4   # cm^2
    A_tel = 120715.7  # cm^2

    # Number of spatial pixels contributing to one spectral pixel
    # i.e., the projected fibre-width on the detector
    N_pix = 6

    # Source flux in units of photons/s/cm2/A
    hc = 1.9865e-08
    photon_flux = temp_flux * temp_wl / hc

    # Calculate each arm separately and stitch at the end:
    var_all = list()
    flux_all = list()
    qual_all = list()
    for arm_num, arm in enumerate(all_arms):
        # Look up the skymodel for the given arm:
        sky = sky_model[arm]  # in units of electrons
        wl = sky['WAVE']

        # Instrument, atmosphere, telescope + fibre efficiencies:
        Q_eff = throughput[arm]

        # Interpolate source flux to sky grid:
        source = np.interp(wl, temp_wl, photon_flux)
        pixel_size = np.diff(wl)
        pixel_size = np.append(pixel_size, pixel_size[-1])

        # Spectral Quantum Factor (empirical from ESO ETC):
        SQF = 1./np.sqrt(2)

        source = source * A_tel * pixel_size * t_exp * Q_eff * SQF
        sky = sky['FLUX'] * t_exp

        # Insert cosmic rays:
        l_CR = CR_rate_per_sec * t_exp * len(wl)
        N_CR = np.random.poisson(l_CR)
        CR_index = np.random.choice(np.arange(len(wl)), N_CR, replace=False)
        source[CR_index] = 10**np.random.normal(3.9, 0.2, N_CR)
        qual = np.zeros_like(wl)
        qual[CR_index] = 1

        # snr = source / np.sqrt(source + sky + N_pix*RON[arm]**2)
        noise = np.sqrt(source + sky + N_pix*RON[arm]**2)
        sensitivity = A_tel * pixel_size * t_exp * Q_eff * SQF * wl / hc
        err_arm = np.interp(wl_joint, wl, noise/sensitivity, left=np.nan, right=np.nan)
        flux_arm = np.interp(wl_joint, wl, source/sensitivity, left=np.nan, right=np.nan)
        qual_arm = np.interp(wl_joint, wl, qual, left=0, right=0)
        qual_arm = 1*(qual_arm > 0)
        var_all.append(err_arm**2)
        flux_all.append(flux_arm)
        qual_all.append(qual_arm)
    flux_all = np.array(flux_all)
    var_all = np.array(var_all)
    flux_joint = np.nansum(flux_all/var_all, axis=0) / np.nansum(1./var_all, axis=0)
    flux_joint = flux_joint * transmission
    err_joint = np.sqrt(1./np.nansum(1./var_all, axis=0))
    qual_joint = 1*(np.sum(qual_all, axis=0) > 0)
    noise = np.random.normal(0., 1., N_pix_in_spectrum)
    flux_joint = flux_joint + noise*err_joint

    # Package Spectrtum:
    if filename:
        save_mock_spectrum(filename, wl_joint, flux_joint, err_joint, qual_joint, t_exp)

    return wl_joint, flux_joint, err_joint, qual_joint


def main():

    import argparse

    parser = argparse.ArgumentParser(description="Simulate a quasar spectrum and apply 4MOST noise")

    parser.add_argument('-z', '--redshift', type=float, default=2.5,
                        help="Redshift")
    parser.add_argument('-G', '--magnitude', type=float, default=21.,
                        help="Gaia magnitude (Vega), roughly equivalenet to SDSS r band")
    parser.add_argument('-t', '--texp', type=float, default=1200.,
                        help="Exposure time in seconds")
    parser.add_argument('--ebv', type=float, default=0.,
                        help="Galactic Extinction, E(B-V)")
    parser.add_argument('--qso-dust', type=float, default=0.,
                        help="Intrinsic A(V) in the quasar rest-frame (SMC-type)")
    parser.add_argument('--sky', type=str, default='dark', choices=['dark', 'grey'],
                        help="Sky brightness, either dark (FLI=0.2) or grey (FLI=0.5)  [default:dark]")
    parser.add_argument('--airmass', type=str, default='mid', choices=['high', 'mid', 'low'],
                        help="Airmass, either high (1.45), mid (1.2) or low (1.05)  [default:mid]")
    parser.add_argument('--lya', action='store_true',
                        help="Include a random Lyman-alpha forest realization (including DLAs)")
    parser.add_argument('--narrow-lines', action='store_true',
                        help="Include additional narrow-line template")
    parser.add_argument('-o', '--output', type=str, default='qso_model.fits',
                        help="Filename of simulated noisy spectrum (FITS Table)")
    parser.add_argument('--model-fname', type=str, default='',
                        help="Filename of noiseless model spectrum (FITS Table)")
    parser.add_argument('--lmin', type=float, default=5900,
                        help="Minimum wavelength of range in which to calculate median signal-to-noise ratio")
    parser.add_argument('--lmax', type=float, default=6100,
                        help="maximum wavelength of range in which to calculate median signal-to-noise ratio")
    parser.add_argument('--dl', type=float, default=1,
                        help="wavelength interval for SNR calculation, i.e., result is given as SNR per dl  [default: 1 Å]")
    
    args = parser.parse_args()

    # Create model template
    temp = make_quasar_template(args.redshift, args.magnitude,
                                Av=args.qso_dust,
                                Ebv=args.ebv,
                                lya_forest=args.lya,
                                narrow_lines=args.narrow_lines,
                                filename=args.model_fname)
    temp_wl, temp_flux, DLA_list = temp

    # Create Mock Observation
    airmass = args.airmass
    moon_phase = args.sky
    transmission = get_transmission(airmass=airmass)
    sky_model = get_sky_model(airmass, moon_phase)
    throughput = get_efficiency(airmass)
    wl, flux, err, qual = apply_noise(temp_wl, temp_flux, sky_model, throughput, transmission,
                                      t_exp=args.texp,
                                      filename=args.output
                                      )

    # Calculate median SNR
    cut = (wl > args.lmin) & (wl < args.lmax)
    SNR = flux / err
    SNR_pix = np.nanmedian(SNR[cut])
    npix = args.dl / 0.25
    SNR_dl = SNR_pix * np.sqrt(npix)

    print("\n 4MOST Quasar Simulator")
    print(" written by J.-K. Krogager, 2020")
    print("--------------------------------")
    print(" z = %.1f" % args.redshift)
    print(" G = %.1f mag" % args.magnitude)
    print(" A(V)_QSO = %.1f mag" % args.qso_dust)

    print(" E(B-V) = %.2f mag  (Rv=3.1)" % args.ebv)
    print(" Exposure time  :  %.1f sec" % args.texp)
    print(" Airmass        :  %.1f" % airmass_conversion[airmass])
    print(" Sky brightness :  %s" % args.sky)
    print(" Seeing         :  0.8 arcsec")

    if args.lya:
        print("Including random Lyman-alpha forest model (incl. DLAs)")
    if args.narrow_lines:
        print("Including narrow emission line model")

    print("Saving simulated spectrum: %s" % args.output)
    if args.model_fname:
        print("Saving noiseless model spectrum: %s" % args.model_fname)
    print("")
    print(" Median signal-to-noise per %.1f Angstrom :  %.1f" % (args.dl, SNR_dl))
    print(" Calculated over wavelength range %.1f -- %.1f Angstrom" % (args.lmin, args.lmax))
    print("")


if __name__ == '__main__':
    main()


