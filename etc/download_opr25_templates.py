import urllib.request
from astropy.table import Table

catalog_fname = '/Users/krogager/Projects/4MOST/opr2.5/exgal_cat_opr25.fits'
cat = Table.read(catalog_fname)

print("downloading templates:")
log = list()
for temp_fname in cat['TEMPLATE']:
    if temp_fname in log:
        continue
    temp_url = 'http://4most.mpe.mpg.de/ETC_spectral_templateInput/OpR/'+temp_fname
    urllib.request.urlretrieve(temp_url, temp_fname)
    log.append(temp_fname)
    print(temp_fname)

