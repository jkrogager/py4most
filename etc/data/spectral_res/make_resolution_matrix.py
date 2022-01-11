import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.interpolate import UnivariateSpline as spline

from astropy.io import fits
from astropy.table import Table


fname_J = '/Users/krogager/Projects/CRS/S8_singlespec/20200907/singlespec/qmost_00033549-2802582_20200907_100009_LJ1.fits'

fnames = {'BLUE': '/Users/krogager/Projects/CRS/S8_singlespec/20200907/singlespec/qmost_00033549-2802582_20200907_100009_LB1.fits',
          'GREEN': '/Users/krogager/Projects/CRS/S8_singlespec/20200907/singlespec/qmost_00033549-2802582_20200907_100009_LG1.fits',
          'RED': '/Users/krogager/Projects/CRS/S8_singlespec/20200907/singlespec/qmost_00033549-2802582_20200907_100009_LR1.fits',
          }

resolution = {}
for arm, fname in fnames.items():
    l, R = np.loadtxt('LR_%s.data' % arm.lower(), unpack=True)
    with fits.open(fname) as hdu:
        tab = hdu[1].data
        wl = tab['WAVE'][0]
    pixsize = np.mean(np.diff(wl))
    R_interp = spline(l, R, s=0.)
    res_pixel = R_interp(wl) / wl / pixsize / 2.355
    resolution[arm] = (wl, res_pixel)
    res_tab = Table({'WAVE': wl, 'R': R_interp(wl), 'WLDISP_PIX': res_pixel}, units=['Angstrom', '', 'pixel'])
    res_tab.write("LR_resolution_%s.fits" % arm, format='fits', overwrite=True)

# -- Load Joint Wavelength Grid:
with fits.open(fname_J) as hdu:
    tab = hdu[1].data
    wl = tab['WAVE'][0]

#wd = list()
#for l, R in resolution.values():
#    R_int = np.interp(wl, l, R, left=np.nan, right=np.nan)
#    wd.append(R_int)
#wd = np.nanmean(wd, axis=0)

wd_max = np.max([np.max(resol) for _, resol in resolution.values()])
nbins = len(wl)
ndiag = int(6*np.ceil(wd_max)+1)

reso = list()

for l, wdisp in resolution.values():
    wd = np.interp(wl, l, wdisp, left=np.nan, right=np.nan)
    r = np.ones([ndiag, nbins])*np.nan
    y0 = ndiag//2
    y = np.arange(ndiag) - y0
    for i, sig in enumerate(wd):
        if sig != np.nan:
            lsf = np.exp(-0.5 * y**2 / sig**2)
            r[:, i] = lsf
    reso.append(r)
reso = np.nanmean(reso, axis=0)
reso /= np.sum(reso, axis=0)
hdr = fits.Header()
hdr['AUTHOR'] = 'JK KROGAGER'
hdr['COMMENT'] = 'Spectral LSF per wavelength bin'
fits.writeto('specres_matrix_4most_LR.fits', data=reso, header=hdr,
             overwrite=True)

