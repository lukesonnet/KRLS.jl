using Distances

# Main KRLS function, takes a 2-dimensional array and a vector, for now
# requires fixed lambda
function krls(Xinit, yinit; lambda = "empty")
  Xinit = float(Xinit)

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
    X[:, i] = X[:, i] / Xinit_sd[i]
  end
  y = (y - yinit_mean) / yinit_sd

  K = gausskernel(X, sigma)

  if lambda == "empty"
    lambda = lambdasearch(y, K)
  end

  out = solveforc(y, K, lambda)
  coeffs = out[1]
  yfitted = K * coeffs

  resid = y - yfitted
  sigmasq = (1/n) .* (resid' * resid)
  vcovmatc = sigmasq .* (K + eye(size(K, 1)) .* lambda)^-2
  vcovmatyhat = K' * (vcovmatc * K)

  #for i in 1:d
  #  for j in 1:n
      #
  #rows = hcat(repmat(reshape(collect(1:n), 1, n), n, 1)[:], repmat(collect(1:n), n, 1)[:])
  return yfitted * yinit_sd + yinit_mean
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
