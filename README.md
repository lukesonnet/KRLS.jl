# KRLS in Julia

This script (very much under construction) implements Kernel Regularized Least Squares [(paper here)](http://www.stanford.edu/~jhain/Paper/PA2014a.pdf) in the Julia language. While much more is on the way, it is currently mostly a transliteration from the R package [(found here)](https://cran.r-project.org/web/packages/KRLS/). It is also currently about 10 times faster!

## Functions

### `krls`

#### Arguments

Required
* `Xinit` - your data matrix, observations are the first dimension, features are the second dimension
* `yinit` - your response vector
Optional
* `lambda` - default is to fit using leave-one-out-cross-validation, but the user can specify any number greater than 0

#### Returns

The method only outputs one object, a `KRLS` object, from which the following can be retrieved:
* `K` - the kernel matrix
* `coeffs` - the choice coefficients
* `Looe` - the final leave-one-out error
* `fitted` - fitted `y` values
* `X` - original `X` matrix
* `y` - original `y` vector
* `sigma`
* `lambda`
* `R2` - R-squared
* `derivatives` - the pointwise marginal effects
* `avgderivatives` - the average pointwise marginal effects
* `var_avgderivatives` - variance of the average pointwise marginal effects
* `vcov_c` - variance covariance matrix of the choice coefficients
* `vcov_fitted` - variance covariance matrix of the fitted values

### `predict`

This can be used to predict outcomes for new data given a new data matrix with the same number of columns.

#### Arguments
* `k` - a KRLS object
* `newmatint` a new data matrix with the same number of columns

#### Returns
Returns a 3-tuple with the following objects (this will be changed soon):
* `yfitted` - predicted `y` values
* `sefit` - standard errors of the predicted values
* `vcovfit` - variance covariance matrix of the predicted values

## Example

```julia
X = randn(1000, 3)
X = hcat(X, vcat(repmat([1], 500, 1), repmat([0], 500, 1)))
y = X * [1,2,3, -2] + randn(1000)
k = krls(X, y)
```




    KRLS results
    ------------------------------------
    Average Marginal Effects:
    1x4 Array{Float64,2}:
     1.0059  1.9553  2.8991  -1.9689
    Quartiles of Marginal Effects
    Var 1: [0.828,1.0257,1.1908]
    Var 2: [1.7346,2.0057,2.2355]
    Var 3: [2.6737,2.91,3.1563]
    Var 4: [-2.207,-2.0203,-1.7358]





```julia
k.coeffs
```




    1000-element Array{Float64,1}:
     -2.10438   
      2.16206   
      1.32769   
      2.9391    
     -2.04586   
     -3.5121    
     -2.07879   
      3.32383   
      2.01171   
      1.10001   
     -2.06835   
     -4.77452   
      0.00034328
      ⋮         
     -2.5586    
     -1.11627   
     -1.32929   
      1.04863   
     -1.33211   
      2.51199   
      1.45449   
      2.41898   
     -1.9107    
      1.75867   
     -2.39654   
     -0.205217  




```julia
hcat(y, k.fitted)
```




    1000x2 Array{Float64,2}:
     -6.28847   -5.4086  
     -2.36659   -3.27057
      0.206222  -0.348901
     -1.76807   -2.99694
     -0.361033   0.494366
     -4.45088   -2.98243
     -1.57046   -0.701293
      1.97831    0.588583
      3.69702    2.8559  
     -6.09572   -6.55565
     -3.53623   -2.67143
      0.304973   2.30125
     -4.1711    -4.17125
      ⋮                  
      5.04699    6.11677
      5.15348    5.62021
     -5.28853   -4.73274
      6.66882    6.23037
      1.31033    1.8673  
     -2.07486   -3.12515
      0.478735  -0.129404
      3.381      2.3696  
     -0.208332   0.590553
      6.1462     5.41089
     -3.31874   -2.31672
     -4.0134    -3.9276  

## Performance

Later I will include more formal speed tests comparing this with KRLS in R and Stata. For now, this (mostly) transliteration of the R package results in speed improvements of 10-20 times.

## To do
* Input validation, more user feedback
* Column names, possibly use DataFrames
* Methods for interpretation
* Cleaning up code
* Speed improvements
