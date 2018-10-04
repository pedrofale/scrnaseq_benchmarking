import numpy as np
from scipy.special import factorial, psi, digamma, polygamma, gammaln
from sklearn.neighbors import NearestNeighbors

def generate_U(N, K, C=2, alpha=1., eps=5.):
	U = sample_gamma(alpha, 1., size=(K, N))
	clusters = np.zeros((N,))
	for c in range(C):
		clusters[int(c*N/C):int((c+1)*N/C)] = clusters[int(c*N/C):int((c+1)*N/C)] + c
		size = U[int(c*K/C):int((c+1)*K/C), int(c*N/C):int((c+1)*N/C)].shape
		U[int(c*K/C):int((c+1)*K/C), int(c*N/C):int((c+1)*N/C)] = sample_gamma(alpha + eps, 1., size=size)
	return U, clusters

def generate_V(P, K, noisy_prop=0., M=2, beta=4., eps=4.):
	P_0 = int((1. - noisy_prop) * P)

	V = np.zeros((P, K))

	if noisy_prop > 0.:
		# noisy genes
		size = V[(P-P_0):, :].shape
		V[(P-P_0):, :] = sample_gamma(0.7, 1, size=size)

	# ungrouped genes
	V[:P_0, :] = sample_gamma(beta, 1, size=(P_0, K))

	# grouped genes
	for m in range(M):
		size = V[int(m*P_0/M):int((m+1)*P_0/M), int(m*K/M):int((m+1)*K/M)].shape
		V[int(m*P_0/M):int((m+1)*P_0/M), int(m*K/M):int((m+1)*K/M)] = sample_gamma(beta + eps, 1., size=size)
	return V

def generate_data(N, P, K, U=None, C=2, alpha=1., eps=5., shape=2., rate=2., zero_prob=0.5, return_all=False):
	if U is None:
		U, clusters = generate_U(N, K, C, alpha, eps)
	else:
		K = U.shape[0]
		N = U.shape[1]

	V = sample_gamma(shape, rate, size=(K, P))
	R = np.matmul(U.T, V)
	X = np.random.poisson(R)

	D = sample_bernoulli(p=1-zero_prob, size=(N, P))
	Y = np.where(D == 0, np.zeros((N, P)), X)

	if return_all:
		return Y, D, X, R, V, U, clusters
	else:
		return Y

def generate_sparse_data(N, P, K, U=None, C=2, alpha=1., eps_U=5., V=None, M=2, beta=4., eps_V=4., noisy_prop=0., zero_prob=0.5, return_all=False):
	if U is None:
		U, clusters = generate_U(N, K, C, alpha, eps_U)
	else:
		K = U.shape[0]
		N = U.shape[1]

	assert K > M

	if V is None:
		V = generate_V(P, K, noisy_prop, M, beta, eps_V)

	R = np.matmul(U.T, V.T) # KxN X PxK
	X = np.random.poisson(R)

	D = sample_bernoulli(p=1-zero_prob, size=(N, P))
	Y = np.where(D == 0, np.zeros((N, P)), X)

	if return_all:
		return Y, D, X, R, V, U, clusters
	else:
		return Y

def generate_scaled_data(N, P, K, U=None, C=2, alpha=1., eps=5., shape=2., rate=2., zero_prob=0.5, mu_lib=8, var_lib=5,return_all=False):
	if U is None:
		U, clusters = generate_U(N, K, C, alpha, eps)
	else:
		K = U.shape[0]
		N = U.shape[1]

	V = sample_gamma(shape, rate, size=(K, P))

	L = sample_gamma(mu_lib**2 / var_lib, mu_lib / var_lib, size=(N,1))

	U = U.T * L
	R = np.matmul(U, V)
	X = np.random.poisson(R)

	D = sample_bernoulli(p=1-zero_prob, size=(N, P))
	Y = np.where(D == 0, np.zeros((N, P)), X)

	if return_all:
		return Y, D, X, R, V, U, L, clusters
	else:
		return Y

def sample_gamma(shape, rate, size=None):
	return np.random.gamma(shape, 1./rate, size=size)

def sample_bernoulli(p, size=None):
	return np.random.binomial(1., p, size=size)

def log_likelihood_L_batches(X, est_U, est_V, est_p_D, est_S, est_L, B, clip=False):
	# Using numerically stable expression for Poisson likelihood
	ll = np.zeros(X.shape)
	
	est_V = est_V * np.concatenate((est_S, np.ones((est_V.shape[0], B.shape[1]))), axis=1)

	est_U = np.concatenate((est_U, B), axis=1) # N x (K + n_batches)
	param = np.dot(est_U, est_V.T)
	param = param * est_L[:, np.newaxis]

	idx = (X != 0)
	ll[idx] = X[idx] * np.log(param[idx] + 1e-7) - param[idx] - gammaln(X[idx] + 1)

	idx = (X == 0)
	ll[idx] = np.log(1.-est_p_D[idx] + est_p_D[idx] * np.exp(-param[idx]) + 1e-7)

	ll = np.sum(ll, axis=1)
	ll = np.mean(ll)

	return ll	

def log_likelihood_L(X, est_U, est_V, est_p_D, est_S, est_L, clip=False):
	# Using numerically stable expression for Poisson likelihood
	ll = np.zeros(X.shape)
	est_V = est_V * est_S
	param = np.dot(est_U, est_V.T)
	param = param * est_L[:, np.newaxis]

	idx = (X != 0)
	ll[idx] = X[idx] * np.log(param[idx] + 1e-7) - param[idx] - gammaln(X[idx] + 1)

	idx = (X == 0)
	ll[idx] = np.log(1.-est_p_D[idx] + est_p_D[idx] * np.exp(-param[idx]) + 1e-7)

	ll = np.sum(ll, axis=1)
	ll = np.mean(ll)

	return ll	

def log_likelihood(X, est_U, est_V, est_p_D, est_S, clip=False):
	# Using numerically stable expression for Poisson likelihood
	ll = np.zeros(X.shape)
	est_V = est_V * est_S
	param = np.dot(est_U, est_V.T)
	
	idx = (X != 0)
	ll[idx] = X[idx] * np.log(param[idx] + 1e-7) - param[idx] - gammaln(X[idx] + 1)

	idx = (X == 0)
	ll[idx] = np.log(1.-est_p_D[idx] + est_p_D[idx] * np.exp(-param[idx]) + 1e-7)

	ll = np.sum(ll, axis=1)
	ll = np.mean(ll)

	return ll	

# def log_likelihood(X, est_U, est_V, est_p_D, est_S, clip=False):
# 	""" Computes the log-likelihood of the model from the inferred latent variables.
# 	"""
# 	N = X.shape[0]

# 	ll = np.zeros(X.shape)

# 	est_V = est_V * est_S
# 	param = np.dot(est_U, est_V.T)
	
# 	idx = (X != 0)
# 	factor = np.log(factorial(X[idx]))
# 	if clip:
# 		factor = X[idx]
# 	ll[idx] = X[idx] * np.log(param[idx] + 1e-7) - param[idx] - factor
	
# 	idx = (X == 0)
# 	ll[idx] = np.log(1.-est_p_D[idx] + est_p_D[idx] * np.exp(-param[idx]) + 1e-7)

# 	ll = np.mean(ll)

# 	return ll

def psi_inverse(initial_x, y, num_iter=5):
    """
    Computes the inverse digamma function using Newton's method
    See Appendix c of Minka, T. P. (2003). Estimating a Dirichlet distribution.
    Annals of Physics, 2000(8), 1-13. http://doi.org/10.1007/s00256-007-0299-1 for details.
    """

    # initialisation
    if y >= -2.22:
        x_old = np.exp(y)+0.5
    else:
        gamma_val = -psi(1)
        x_old = -(1/(y+gamma_val))

    # do Newton update here
    for i in range(num_iter):
        numerator = psi(x_old) - y
        denumerator = polygamma(1, x_old)
        x_new = x_old - (numerator/denumerator)
        x_old = x_new

    return x_new

def entropy_batch_mixing(latent_space, batches):
    def entropy(hist_data):
        n_batches = len(np.unique(hist_data))
        if n_batches > 2:
            raise ValueError("Should be only two clusters for this metric")
        frequency = np.mean(hist_data == 1)
        if frequency == 0 or frequency == 1:
            return 0
        return -frequency * np.log(frequency) - (1 - frequency) * np.log(1 - frequency)

    nne = NearestNeighbors(n_neighbors=51, n_jobs=8)
    nne.fit(latent_space)
    kmatrix = nne.kneighbors_graph(latent_space) - scipy.sparse.identity(latent_space.shape[0])

    score = 0
    for t in range(50):
        indices = np.random.choice(np.arange(latent_space.shape[0]), size=100)
        score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]\
                                 [kmatrix[indices].nonzero()[0] == i]]) for i in range(100)])
    return score / 50.

# def imputation_error(true, imputed, dropout_idx):
# 	"""
# 	Computes the median L1 distance between the original count before dropout and the 
# 	imputed value.
# 	"""
# 	return np.median(np.abs(true[dropout_idx] - imputed[dropout_idx]))

# def make_dropouts(X, dropout_rate=0.1):
#     """
#     Romain Lopez

#     X: original testing set
#     ========
#     returns:
#     X_corrupted: copy of X with zeros
#     i, j, ix: indices of where dropout is applied
#     """
#     X_corrupted = np.copy(X)
#     D = np.ones(X.shape)

#     # select non-zero subset
#     i,j = np.nonzero(X_corrupted)
    
#     # choice number 1 : select 10 percent of the non zero values (so that distributions overlap enough)
#     ix = np.random.choice(range(len(i)), int(np.floor(0.1 * len(i))), replace=False)
#     D[i[ix], j[ix]] = sample_bernoulli(dropout_rate) # D contains the entries to which dropout may be applied (if 0, it's a dropout)
    
#     X_corrupted[i[ix], j[ix]] *= np.random.binomial(1, dropout_rate) # 1-dropout_rate * 100 % of those entries are actually dropped out
    
#     return X_corrupted, D

def dropout(X, rate=0.1):
    """
    X: original testing set
    ========
    returns:
    X_zero: copy of X with zeros
    i, j, ix: indices of where dropout is applied
    """
    X_zero = np.copy(X)
    # select non-zero subset
    i,j = np.nonzero(X_zero)
    
    # choice number 1 : select 10 percent of the non zero values (so that distributions overlap enough)
    ix = np.random.choice(range(len(i)), int(np.floor(0.1 * len(i))), replace=False)
    X_zero[i[ix], j[ix]] *= np.random.binomial(1, rate)
       
    corruption_info = {'X_corr': X_zero, 'i': i, 'j': j, 'ix': ix}
    # choice number 2, focus on a few but corrupt binomially
    #ix = np.random.choice(range(len(i)), int(slice_prop * np.floor(len(i))), replace=False)
    #X_zero[i[ix], j[ix]] = np.random.binomial(X_zero[i[ix], j[ix]].astype(np.int), rate)
    return corruption_info

def imputation_error(X_mean, X, X_zero, i, j, ix):
    """
    X_mean: imputed dataset
    X: original dataset
    X_zero: zeros dataset
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    median L1 distance between datasets at indices given
    """
    all_index = i[ix], j[ix]
    x, y = X_mean[all_index], X[all_index]
    return np.median(np.abs(x - y))