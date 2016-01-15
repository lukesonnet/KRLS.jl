using Distances

# Main KRLS function, takes a 2-dimensional array and a vector, for now
# requires fixed lambda
function krls(Xinit, yinit; lambda = 0.5)
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

  out = solveforc(y, K, lambda)

  return K * out * yinit_sd + yinit_mean
end

# Computes a Kernel matrix using the Gaussian Kernel
function gausskernel(X, sigma)
  return exp(-1 .* pairwise(Euclidean(), X').^2 ./ sigma)
end

# Solve for the choice coefficients
function solveforc(y, K, lambda)
  return inv(K + eye(size(K, 1)) .* lambda) * y
end
