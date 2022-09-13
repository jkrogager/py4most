#!/usr/local/opt/python@3.8/bin/python3.8
"""
Spectral simulator for 4MOST LRS

    Note: Current version generates a subset of 10% from the input catalog
    and adds 1000 high redshift Type I quasars from z=4 to z=6.

The spectra are generated based on a set of spectral templates located in the folder
    TEMP_PATH

The code takes as input the input catalog from 4FS (facility simulator) which for each
target specifies RA, DEC, TEMPLATE, REDSHIFT, MAGNITUDE, MAG_FILTER, TRG_UID, CNAME

`TEMPLATE` here refers to a filename which must be present in the `TEMP_PATH` folder.
These templates should follow the format for 4FS templates: LAMBDA, FLUX_DENSITY
Templates are then redshifted and normalized to the given input magnitude.
    (Note that if templates are already redshifted, they will not be redshifted again.
    This is determined from the FITS header. If EXTNAME='SPECTRUM' then a redshift will
    be applied. Otherwise the template is assumed to be shifted already.)

Optionally, a random Lyman-alpha forest realisation can be applied to sources at z > 2.
Cosmic rays can also be included, which will add a random number of cosmics based on
statistics of the incidence rate on the CCD per second which is then summed up over 5
rows (to mimic the L1 extraction).

The throughput is calculated in two bins of sky brightness:
    dark : FLI = 0.2
    grey : FLI = 0.5

and three bins of airmass:
    low  : 1.05
    mid  : 1.20
    high : 1.45

The transmission curves are tabulated using the 4MOST ETC:
https://etc.eso.org/observing/etc/fourmost

The output is written to the folder: OUTPUT_DIR

"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d
from astropy.io import fits
from astropy.table import Table

import os
import sys

from py4most.etc.dust import MW_reddening
from py4most.etc import lya


# -- Verify data and template paths:
TEMP_PATH = '/Users/krogager/Projects/4MOST/templates/opr25'
OUTPUT_DIR = '/Users/krogager/Projects/4MOST/test_data'

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'data/')



## -- AIRMASS
airmass_conversion = {
        'high': 1.45,
        'mid': 1.2,
        'low': 1.05,
        }

## -- MOON_PHASE
sky_conversion = {
        'dark': 0.2,
        'grey': 0.5,
        }

# Gaia G-band filter curve:
G_filter = np.loadtxt(data_path+'Gaia_G.tab')
r_filter = np.loadtxt(data_path+'SDSS_r.tab')
z_filter = np.loadtxt(data_path+'SDSS_z.tab')
filters = {
        'G': G_filter,
        'r': r_filter,
        'z': z_filter,
        }

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


def synthetic_band(wavelength, flux, band='G'):
    filter_curve = filters[band]
    T = np.interp(wavelength, filter_curve[:, 0], filter_curve[:, 1], left=0, right=0)
    synphot = np.sum(flux*T)/np.sum(T)
    wl_band = np.sum(wavelength*T)/np.sum(T)
    return synphot, wl_band


def load_template(filename, redshift, mag, band, Ebv=0, lya_forest=True):
    h = fits.getheader(filename, 1)
    data = Table.read(filename)

    if h.get('EXTNAME', '') == 'SPECTRUM':
        z = redshift
    elif 'REDSHIFT' in h:
        redshift = h['REDSHIFT']
        z = 0.
    elif 'zmin_35' in filename:
        z = 4.
    elif 'zmin_45' in filename:
        z = 5.25
    else:
        z = 0.

    # Redshift and Normalize Template:
    temp_wl = data['LAMBDA'] * (z+1)
    temp_flux = data['FLUX_DENSITY']
    f0, wl_band = synthetic_band(temp_wl, temp_flux, band)
    f_band = 1./(wl_band)**2 * 10**(-(mag+2.406)/2.5)
    temp_flux = temp_flux/f0*f_band

    if Ebv > 0:
        # assume average Rv = 3.1
        Av_gal = Ebv * 3.1
        temp_flux = temp_flux * 10**(-0.4*ksi_mw*Av_gal)

    if lya_forest:
        if z > 2.1:
            wl_lya, T_lya, DLA_list = lya.lya_transmission(2.0, z, R=4000.)
            lya_profile = np.interp(temp_wl, wl_lya, T_lya, right=1., left=0.)
            temp_flux = temp_flux * lya_profile

    return temp_wl, temp_flux, redshift


def save_mock_spectrum(filename, wl, flux, err, qual, t_exp, ra, dec, cname, uid,
                       redshift=None, magnitude=None, mag_filter=None, tempname=''):
    N = len(wl)
    hdu = fits.HDUList()

    hdr = fits.getheader(data_path + 'temp_header.fits')
    hdr_tab = fits.getheader(data_path + 'temp_header.fits', 1)
    hdr['EXPTIME']  = (t_exp, 'Total integration time per data element (s)')
    hdr['RA'] = ra
    hdr['DEC'] = dec
    hdr_tab['CNAME'] = cname
    hdr_tab['TRG_NME'] = cname
    hdr_tab['TRG_UID'] = uid
    hdr['CNAME'] = cname
    hdr['TRG_UID'] = uid
    if redshift is not None:
        hdr_tab['REDSHIFT'] = redshift
    if magnitude is not None:
        hdr_tab['MAG'] = magnitude
    if mag_filter is not None:
        hdr_tab['MAG_FILT'] = mag_filter
    if tempname:
        hdr_tab['TEMPNAME'] = tempname
    SNR = np.nanmedian(flux/err)
    if not np.isfinite(SNR):
        return False
    hdr_tab['SNR'] = SNR
    hdr['SNR'] = SNR

    prim = fits.PrimaryHDU(header=hdr)
    hdu.append(prim)

    pre = '%i' % len(wl)
    tab = fits.BinTableHDU.from_columns([
                    fits.Column(name='WAVE', array=np.array([wl]),
                                format=pre+'E', unit='Angstrom', disp='F9.4'),
                    fits.Column(name='FLUX', array=np.array([flux]),
                                format=pre+'E', unit='erg/(s*cm^2*Angstrom)', disp='E13.5E2'),
                    fits.Column(name='ERR_FLUX', array=np.array([err]),
                                format=pre+'E', unit='erg/(s*cm^2*Angstrom)', disp='E13.5E2'),
                    fits.Column(name='QUAL', array=np.array([qual]),
                                format=pre+'J', null=-1, disp='I1'),
                    fits.Column(name='FLUX_NOSS', array=np.array([flux*0.]),
                                format=pre+'E', unit='erg/(s*cm^2*Angstrom)', disp='E13.5E2'),
                    fits.Column(name='ERR_FLUX_NOSS', array=np.array([err*0.]),
                                format=pre+'E', unit='erg/(s*cm^2*Angstrom)', disp='E13.5E2'),
                    ], header=hdr_tab)
    tab.name = 'PHASE3SPECTRUM'
    hdu.append(tab)

    hdu.writeto(filename, overwrite=True, output_verify='silentfix')


def apply_noise(temp_wl, temp_flux, sky_model, throughput, transmission,
                t_exp=1200, ra=0, dec=0, cname='', uid=1, cr=True, redshift=None,
                magnitude=None, mag_filter=None, tempname='', output_dir='output'):
    """
    temp_wl : array of wavelengths in the input template (units: Angstrom)
    temp_flux : array of fluxes in the input template (units: erg/s/cm2/A)
    sky_model : FITS-Table with three extensions {BLUE, GREEN, RED}
                The table contains the following columns: WAVE, FLUX
    throughput : FITS-Table of effective throughput including instrument, fibre and telescope
                 with three extensions {BLUE, GREEN, RED}
    transmission : array of atmospheric transmission model (telluric absorption).
                   Other absorption features can be included as well, such as metal abs lines.
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
        qual = np.zeros_like(wl)
        if cr:
            l_CR = CR_rate_per_sec * t_exp * len(wl)
            N_CR = np.random.poisson(l_CR)
            CR_index = np.random.choice(np.arange(len(wl)), N_CR, replace=False)
            source[CR_index] = 10**np.random.normal(3.9, 0.2, N_CR)
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
    fibnum = np.random.randint(1, 812)
    # fibnum = 101
    filename = os.path.join(output_dir, cname+'_20221010_1%05i_LJ1.fits' % fibnum)
    save_mock_spectrum(filename, wl_joint, flux_joint, err_joint, qual_joint, t_exp,
                       ra=ra, dec=dec, cname=cname, uid=uid, redshift=redshift,
                       magnitude=magnitude, mag_filter=mag_filter, tempname=tempname)

    return wl_joint, flux_joint, err_joint, qual_joint


def decimal2string(ra, dec, delimiter=':'):
    # Convert degrees to sexagesimal:
    hour_angle = ra/15.
    hours = np.floor(hour_angle)
    minutes = np.remainder(hour_angle, 1)*60.
    seconds = np.remainder(minutes, 1)*60.
    hms = ["%02.0f"%hours, "%02.0f"%np.floor(minutes), "%05.2f"%seconds]
    ra_str = delimiter.join(hms)
    ra_str = ra_str.replace('.', '')

    sign = np.sign(dec)
    degrees = np.abs(dec)
    arcmin = np.remainder(degrees, 1)*60.
    arcsec = np.remainder(arcmin, 1)*60.
    dms = ["%+02.0f"%(sign*np.floor(degrees)), "%02.0f"%np.floor(arcmin), "%04.1f"%arcsec]
    dec_str = delimiter.join(dms)
    dec_str = dec_str.replace('.', '')

    return (ra_str, dec_str)


def main():

    parser = argparse.ArgumentParser(prog='qmost_etc',
                            description="Simulate 4MOST spectra based on an input catalog")

    parser.add_argument('input', type=str,
                        help='Filename of FITS catalog')
    parser.add_argument('-t', '--texp', type=float, default=1200.,
                        help="Exposure time in seconds")
    parser.add_argument('--sky', type=str, default='dark', choices=['dark', 'grey'],
                        help="Sky brightness: dark (FLI=0.2) or grey (FLI=0.5)  [default:dark]")
    parser.add_argument('--airmass', type=str, default='mid', choices=['high', 'mid', 'low'],
                        help="Airmass, either high (1.45), mid (1.2) or low (1.05)  [default:mid]")
    parser.add_argument('--cr', action='store_true',
                        help="Include cosmic rays")

    args = parser.parse_args()

    # Load target catalog:
    cat = Table.read(args.input)

    # Get ambient parameters:
    airmass = args.airmass
    moon_phase = args.sky
    transmission = get_transmission(airmass=airmass)
    sky_model = get_sky_model(airmass, moon_phase)
    throughput = get_efficiency(airmass)

    for row in cat:
        temp_fname = os.path.join(TEMP_PATH, row['TEMPLATE'])
        if 'SN' in temp_fname:
            continue
        mag = row['MAGNITUDE']
        redshift = row['REDSHIFT']
        band = 'r' if 'r' in row['MAG_FILTER'] else 'z'
        reddening = 0.

        temp_wl, temp_flux, redshift = load_template(
                                           temp_fname,
                                           redshift,
                                           mag, band,
                                           Ebv=reddening,
                                           lya_forest=False,
                                           )

        # Create Mock Observation
        wl, flux, err, qual = apply_noise(temp_wl, temp_flux, sky_model, throughput, transmission,
                                          t_exp=args.texp,
                                          cr=args.cr,
                                          ra=row['RA'],
                                          dec=row['DEC'],
                                          cname=row['CNAME'],
                                          uid=row['TRG_UID'],
                                          redshift=redshift,
                                          magnitude=mag,
                                          mag_filter=row['MAG_FILTER'],
                                          tempname=row['TEMPLATE'],
                                          output_dir=OUTPUT_DIR,
                                          )


if __name__ == '__main__':
    main()


