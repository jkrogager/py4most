
A collection of scripts to generate 4MOST like L1 spectra

The general ETC takes a FITS catalog as input and produces a random noise realization
for each target in the catalog (incl. cosmic rays and randomized Lya forest).
The parameters can be controlled from the command line. See the help function:

 %] python3 general_etc.py -h

or check the doc-string of the python source file `general_etc.py`.

The first step is to download the spectral templates needed using the script
`download_opr25_templates.py` with the catalog filename as input on the command line:
 %] python3 download_opr25_templates.py  *catalog_fname.fits*

The OpR 2.5 catalogs of galactic and extragalactic targets are available on my Google Drive
under folder 4MOST/IWG8.

The path to these templates should then be inserted in the script `general_etc.py`.

