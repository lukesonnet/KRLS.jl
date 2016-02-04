# KRLS in Julia

This script (very much under construction) implements Kernel Regularized Least Squares [(paper here)](http://www.stanford.edu/~jhain/Paper/PA2014a.pdf) in the Julia language. Currently it only returns the fitted $y$ values, although much more can be recovered by tweaking the script. While much more is on the way, it is currently mostly a transliteration from the R package [(found here)](https://cran.r-project.org/web/packages/KRLS/). It is also currently about 10 times faster!

To do:
* Create KRLS type
* Redo pointwise marginal effects for binary variables
* Methods for interpretation
