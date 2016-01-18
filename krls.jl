using Distances

# todo, carry around column names, perhaps change to accept data frame?

type KRLS
  K::Array{Float64, 2}
  coeffs
  Looe
  fitted
  X::Array{Float64, 2}
  y
  sigma
  lambda
  R2::Float64
  derivatives::Array{Float64, 2}
  avgderivatives::Array{Float64, 2}
  var_avgderivatives::Array{Float64, 2}
  vcov_c::Array{Float64, 2}
  vcov_fitted::Array{Float64, 2}
end

# Main KRLS function, takes a 2-dimensional array and a vector, for now
# requires fixed lambda
function krls(Xinit::Array, yinit::Array; lambda = "empty")
  Xinit = float(Xinit)
  yinit = float(yinit)

  n = size(Xinit, 1)
  d = size(Xinit, 2)

  sigma = d

  X = deepcopy(Xinit)
  Xinit_sd = std(X, 1)
  Xinit_mean = mean(X, 1)
  y = deepcopy(yinit)
  yinit_mean = mean(y)
  yinit_sd = std(y)

  for i in 1:size(X, 2)
    X[:, i] = (X[:, i] - Xinit_mean[i]) / Xinit_sd[i]
  end
  y = (y - yinit_mean) / yinit_sd

  K = gausskernel(X, sigma)

  if lambda == "empty"
    lambda = lambdasearch(y, K)
  end

  coeffs, Le = solveforc(y, K, lambda)
  yfitted = K * coeffs

  resid = y - yfitted
  sigmasq = (1/n) .* (resid' * resid)
  vcovmatc = sigmasq .* (K + eye(size(K, 1)) .* lambda)^-2
  vcovmatyhat = K' * (vcovmatc * K)

  derivmat = Array(Float64, n, d)
  avgderivmat = Array(Float64, 1, d)
  varavgderivmat = Array(Float64, 1, d)
  tempL = Array(Float64, n, n)
  # There is a lot of redundant information here (i.e. symmetric matrices)
  for j in 1:d
    for i in 1:n
      tempL[i, :] = (K[:,i] .* (X[i, j] - X[:, j]))
      derivmat[i, j] = (-2 / sigma) .* sum(coeffs .* squeeze(tempL[i, :], 1))
    end
    varavgderivmat[1, j] = (1 / n^2) .* sum((-2/sigma)^2 .* (tempL' * vcovmatc * tempL))
  end
  avgderivmat = mean(derivmat, 1)

  derivmat = yinit_sd .* derivmat
  avgderivmat = yinit_sd .* avgderivmat

  for j in 1:d
    derivmat[:, j] = derivmat[:, j] ./ Xinit_sd[j]
    avgderivmat[j] = avgderivmat[j] ./ Xinit_sd[j]
    varavgderivmat[j] = varavgderivmat[j] .* (yinit_sd ./ Xinit_sd[j])^2
  end

  yfitted = yfitted * yinit_sd + yinit_mean
  vcov_c = vcovmatc * (yinit_sd^2)
  vcov_fitted = vcovmatyhat * (yinit_sd^2)

  Looe = Le * yinit_sd

  R2 = 1 - (var(yinit - yfitted)/(yinit_sd^2))

  k = KRLS(K, coeffs, Looe, yfitted, Xinit, yinit, sigma, lambda, R2,
              derivmat, avgderivmat, varavgderivmat, vcov_c, vcov_fitted)

  # Calculates first differences for binary variables
  #for j in 1:d
  #  if size(unique(Xinit[:, j]), 1) == 2
  #    X0 = deepcopy(Xinit)
  #    X1 = deepcopy(Xinit)
  #    X1[:, j] = maximum(X1[:, j])
  #    X0[:, j] = minimum(X0[:, j])

  return k
end

# Solve for the choice coefficients
function solveforc(y, K, lambda)

  Ginv = inv(K + eye(size(K, 1)) .* lambda)
  coeffs = Ginv * y
  Lepart = coeffs ./ diag(Ginv)

  return coeffs, (Lepart' * Lepart)[1]
end

# Solve for lambda using GCV
function lambdasearch(y, K)
  n = length(y)
  tol = 10.0^-3 * n
  U = deepcopy(n)

  Keig = eigvals(K)

  while sum(Keig / (Keig + U)) < 1
    U -= 1
  end

  q = size(K, 1) + 1 - findmin(abs(Keig - (maximum(Keig)/1000)))[2]

  L = eps(Float64)

  while sum(Keig / (Keig + L)) > q
    L += 0.05
  end

  X1 = L + (.381966)*(U-L)
  X2 = U - (.381966)*(U-L)

  S1 = solveforc(y, K, X1)[2]
  S2 = solveforc(y, K, X2)[2]

  while abs(S1 - S2) > tol

    if S1 < S2
      U  = X2
      X2 = X1
      X1 = L + (.381966)*(U-L)
      S2 = S1
      S1 = solveforc(y, K, X1)[2]
    else #S2 < S1
      L  = X1
      X1 = X2
      X2 = U - (.381966)*(U-L)
      S1 = S2
      S2 = solveforc(y, K, X2)[2]
    end
  end

  if S1 < S2
    return X1
  else
    return X2
  end
end

# Computes a Kernel matrix using the Gaussian Kernel
function gausskernel(X, sigma)
  return exp(-1 .* pairwise(Euclidean(), X').^2 ./ sigma)
end

function Base.show(io::IO,k::KRLS)
  println(io,"KRLS results")
  println(io,"------------------------------------")
  println(io,"Average Marginal Effects:")
  println(io,round(k.avgderivatives, 4))
  println(io,"Quartiles of Marginal Effects")
  for j in 1:size(k.derivatives, 2)
    deriv_quantile = round(quantile(k.derivatives[:, j], [0.25, 0.5, 0.75]), 4)
    println(io, "Var $j: $deriv_quantile")
  end
end

# todo: dimension and name checking here
function predict(k::KRLS, newmatinit::Array)

  n = size(k.X, 1)

  X = deepcopy(k.X)
  Xinit_sd = std(X, 1)
  Xinit_mean = mean(X, 1)
  newmat = float(deepcopy(newmatinit))

  for i in 1:size(X, 2)
    X[:, i] = (X[:, i] - Xinit_mean[i]) / Xinit_sd[i]
    newmat[:, i] = (newmat[:, i] - Xinit_mean[i]) / Xinit_sd[i]
  end

  nn = size(newmat, 1)

  newK = gausskernel(vcat(newmat, X), k.sigma)[1:nn, (nn + 1):(nn + n)]

  yfitted = newK * k.coeffs

  # todo: add standard errors

  yfitted = yfitted .* std(k.y) .+ mean(k.y)

  return yfitted
end
