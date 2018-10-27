from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.base import BaseEstimator
import numpy as np
from scipy.stats import pearsonr
from misc import utils

# class ModelWrapper(object):
# 	def __init__(self, X_train, c_train, D_train=None, X_train=None, X_test=None, c_test=None, name="New model"):
# 		self.X_train = X_train
# 		self.D_train = D_train
# 		self.dropout_idx = None
# 		if D_train is not None:
# 			self.dropout_idx = np.where(self.D_train == 0)
# 		self.X_train = X_train
# 		self.c_train = c_train

# 		self.X_test = X_test
# 		self.c_test = c_test

# 		self.name = name
# 		self.est_z = None
# 		self.est_R = None
# 		self.est_D = None
# 		self.proj_2d = None
# 		self.silhouette = None
# 		self.ari = None
# 		self.nmi = None
# 		self.dropid_acc = None
# 		self.dropimp_err = None

# 	def run(self, run_func, est_real_func=None, est_drop_func=None, do_silh=True, do_tsne=True, do_imp=False):
# 		self.est_z = run_func(self.X_train)

# 		if do_silh:
# 			self.silhouette = silhouette_score(self.est_z, self.c_train)

# 			C = np.unique(self.c_train).size
# 			kmeans = KMeans(n_clusters=C)
# 			res = kmeans.fit_predict(self.est_z)

# 			self.ari = adjusted_rand_score(self.c_train, res)

# 			self.nmi = normalized_mutual_info_score(self.c_train, res)

# 		if self.est_z.shape[1] > 2:
# 			if do_tsne:
# 				self.proj_2d = TSNE(n_components=2).fit_transform(self.est_z)
# 		else:
# 			self.proj_2d = self.est_z

# 		if do_imp:
# 			if self.D_train is not None and est_drop_func is not None:
# 				self.est_D = est_drop_func()
# 				self.dropid_acc = accuracy_score(self.est_D.flatten(), self.D_train.flatten())
# 				if self.X_train is not None and est_real_func is not None:
# 					self.est_R = est_real_func()
# 					self.dropimp_err = utils.imputation_error(self.X_train, self.est_R, self.dropout_idx)

class ModelWrapper(object):
	def __init__(self, model_inst, X_train, c_train=None, b_train=None, X_test=None, b_test=None, do_imp=False, log_data=False, name=None):
		"""
		X_train is raw counts. It can be just the name of a file from which the data will be loaded in batches, if the base algorithm supports it.
		If log_data is True, transform X_train to pseudo log-counts.
		b_train and b_test are 1-D integers, not one-hot.
		"""
		# If it is a BaseEstimator, it has the methods .fit, .transform, .fit_transform and .score
		assert issubclass(model_inst.__class__, BaseEstimator)

		self.model_inst = model_inst

		self.X_train = X_train
		self.c_train = c_train
		self.b_train = b_train
		self.batches = b_train is not None
		if self.batches:
			self.n_batches = b_train.shape[1]

		self.X_test = X_test
		self.b_test = b_test

		self.do_imp = do_imp
		if do_imp:
			self.corruption_info = utils.dropout(self.X_train)

		self.log_data = log_data
		if self.log_data:
			self.X_train = np.log(X_train + 1.)
			if X_test is not None:
				self.X_test = np.log(X_test + 1.)
			if do_imp:
				self.corruption_info['X_corr'] = np.log(self.corruption_info['X_corr'] + 1.)

		if name is None:
			name = self.model_inst.__class__.__name__
		self.name = name
		self.est_z = None
		self.est_mean = None
		self.proj_2d = None
		self.silhouette = None
		self.asw = None
		self.ari = None
		self.nmi = None
		self.dropimp_err = 100000.
		self.train_ll = None
		self.test_ll = None
		self.batch_asw = None

	def run(self, max_iter=1, max_time=60, do_tsne=True, do_dll=True, do_holl=True, do_silh=True, do_batch=False, do_corr=False, verbose=False):
		if verbose:
			print('Running {0}...'.format(self.name))

		X = self.X_train
		if self.do_imp:
			X = self.corruption_info['X_corr']
			
		try:
			self.est_z = self.model_inst.fit_transform(X, batch_idx=self.b_train, max_time=max_time, max_iter=max_iter)
		except TypeError:
			#do stuff
			print('Some arguments were ignored by {}.'.format(self.model_inst.__class__.__name__))
			try:
				self.est_z = self.model_inst.fit_transform(X, max_time=max_time, max_iter=max_iter)
			except TypeError:
				print('Running .fit_transform() without keyword arguments.')
				self.est_z = self.model_inst.fit_transform(X)

		if self.do_imp:
			est_mean = self.model_inst.get_est_X()
			X_original = self.X_train

			if self.log_data:
				est_mean = np.exp(est_mean) - 1 # log to counts
				X_original = np.exp(self.X_train) - 1 # log to counts

			self.dropimp_err = utils.imputation_error(est_mean, X_original, 
				self.corruption_info['X_corr'], self.corruption_info['i'], self.corruption_info['j'], self.corruption_info['ix'])
		else:
			# Only perform any further analysis if data has not been corrupted for imputation testing
			if do_silh and self.c_train is not None:
				self.silhouette = silhouette_score(self.est_z, self.c_train)
				self.asw = self.silhouette
				if self.batches and do_batch:
					if self.n_batches > 2:
						raise ValueError("Only 2 batches supported.")
					self.batch_asw = silhouette_score(self.est_z, self.b_train[:, 0]) # this only works for 2 batches!!!

				C = np.unique(self.c_train).size
				kmeans = KMeans(n_clusters=C, n_init=200, n_jobs=8)
				res = kmeans.fit_predict(self.est_z)

				self.ari = adjusted_rand_score(self.c_train, res)

				self.nmi = normalized_mutual_info_score(self.c_train, res)

			if do_tsne:
				if self.est_z.shape[1] > 2:
					self.do_tsne()
			else:
				self.proj_2d = self.est_z

			if do_dll:
				if verbose:
					print('Evaluating train-data log-likelihood...')
				try:
					self.train_ll = -self.model_inst.loss_dict['t_loss'][-1]
				except AttributeError:
					try:
						self.train_ll = self.model_inst.score(self.X_train, batch_idx=self.b_train)
					except TypeError:
						self.train_ll = self.model_inst.score(self.X_train)
				if self.log_data:
					self.train_ll = self.train_ll - np.mean(np.sum(self.X_train, axis=-1))
			
			if do_holl:
				if self.X_test is not None:
					if verbose:
						print('Evaluating test-data log-likelihood...')
					try:
						if len(self.model_inst.loss_dict['v_loss']) > 0:
							self.test_ll = -self.model_inst.loss_dict['v_loss'][-1]
						else:
							try:
								self.test_ll = self.model_inst.score(self.X_test, batch_idx=self.b_test)
							except TypeError:
								self.test_ll = self.model_inst.score(self.X_test)
					except AttributeError:
						try:
							self.test_ll = self.model_inst.score(self.X_test, batch_idx=self.b_test)
						except TypeError:
							self.test_ll = self.model_inst.score(self.X_test)
					if self.log_data:
						self.test_ll = self.test_ll - np.mean(np.sum(self.X_test, axis=-1))

			if do_corr:
				if self.est_z.shape[1] == 2:
					self.library_size = np.sum(self.X_train, axis=1)
					self.detection_rate = np.sum(self.X_train != 0, axis=1) / self.X_train.shape[1]

					self.corr_1_ls = np.abs(pearsonr(self.est_z[:, 0], self.library_size)[0])
					self.corr_1_dr = np.abs(pearsonr(self.est_z[:, 0], self.detection_rate)[0])
					
					self.corr_2_ls = np.abs(pearsonr(self.est_z[:, 1], self.library_size)[0])
					self.corr_2_dr = np.abs(pearsonr(self.est_z[:, 1], self.detection_rate)[0])

		if verbose:
			print('Done.')

	def do_tsne(self):
		self.proj_2d = TSNE(n_components=2).fit_transform(self.est_z)