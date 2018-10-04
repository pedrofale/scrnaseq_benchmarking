import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import warnings
from rpy2.rinterface import RRuntimeWarning

class pCMF(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10):
        self.est_X = None
        self.est_D = None
        self.n_components = n_components

        warnings.filterwarnings("ignore", category=RRuntimeWarning)
        rpy2.robjects.numpy2ri.activate()
        ro.r["library"]("pCMF")
        ro.r.assign("ncomp", n_components)
        
    def fit_transform(self, X, max_iter=1000, max_time=60.):
        self.fit(X, max_iter=max_iter)
        return self.transform(X)
        
    def fit(self, X, max_iter=1000):
        self.X_ = X
        nr, nc = X.shape
        X_trainr = ro.r.matrix(X, nrow=nr, ncol=nc)
        ro.r.assign("X_train", X_trainr)
        ro.r.assign("max_iter", max_iter)

        ro.r("res <- pCMF(X_train, K=ncomp, verbose=TRUE, sparsity=TRUE, zero_inflation=TRUE, ncores=10, iter_max=max_iter)")

        self.params = ro.r("res")
        self.train_ll_it = ro.r("res$monitor$loglikelihood")
        # Return the classifier
        return self
    
    def transform(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_'])

        # Input validation
        X = check_array(X)
        nr, nc = X.shape
        X_testr = ro.r.matrix(X, nrow=nr, ncol=nc)

        U = ro.r("getU(res, log_representation=FALSE)")
        V = ro.r("getV(res)")
        self.est_X = np.matmul(U, V.T)

        return U

    def score(self, X):
        print('Ghislain Durif\'s package does not contain methods to compute the log-likelihood.')

    def get_est_X(self):
        return self.est_X
