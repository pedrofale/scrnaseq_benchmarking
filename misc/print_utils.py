import operator

def print_full_report(model_list, test_ll=False, cluster=False, dropout=False, batch=False, filename=None, filemode='w'):
	print('\033[1mModel results:\033[0m\n')
	print_model_lls(model_list, mode='Train', filename=filename, filemode=filemode)
	if test_ll:
		print('')
		print_model_lls(model_list, mode='Test', filename=filename, filemode=filemode)
	if cluster:
		print('')
		print_model_clustering(model_list, filename=filename, filemode=filemode)
	if dropout:
		print('')
		print_model_dropimp_errs(model_list, filename=filename, filemode=filemode)
	if batch:
		print('')
		print_model_batch_asws(model_list, filename=filename, filemode=filemode)

def print_model_lls(model_list, mode='Train', filename=None, filemode='w'):
	""" Print ordered train or test log-likelihoods.
	"""
	assert mode in ['Train', 'Test']

	f = None
	if filename is not None:
		f = open(filename, filemode)

	names = []
	lls = []
	for model in model_list:
		names.append(model.name)
		if mode=='Train':
			lls.append(model.train_ll)
		elif mode=='Test':
			assert model.test_ll is not None
			lls.append(model.test_ll)

	scores = dict(zip(names, lls))

	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

	print('{} data log-likelihood:'.format(mode), file=f)
	print('\033[1m- {0}: {1:.6}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]), file=f)
	for score_tp in sorted_scores[1:]:
		print('- {0}: {1:.6}'.format(score_tp[0], score_tp[1]), file=f)

	if f is not None:
		f.close()

def print_model_batch_asws(model_list, filename=None, filemode='w'):
	""" Print ordered batch ASW scores.
	"""
	f = None
	if filename is not None:
		f = open(filename, filemode)

	names = []
	batch_asws = []
	for model in model_list:
		names.append(model.name)
		batch_asws.append(model.batch_asw)

	scores = dict(zip(names, batch_asws))

	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=False) # lower is better

	print('Batch ASW scores:', file=f)
	print('\033[1m- {0}: {1:.6}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]), file=f)
	for score_tp in sorted_scores[1:]:
		print('- {0}: {1:.6}'.format(score_tp[0], score_tp[1]), file=f)

	if f is not None:
		f.close()

def print_model_clustering(model_list, filename=None, filemode='w'):
	""" Print ordered ASW, ARI and NMI scores.
	"""
	print_model_asws(model_list, filename=filename, filemode=filemode)
	print('')
	print_model_aris(model_list, filename=filename, filemode=filemode)
	print('')
	print_model_nmis(model_list, filename=filename, filemode=filemode)

def print_model_aris(model_list, filename=None, filemode='w'):
	""" Print ordered ARI scores.
	"""
	f = None
	if filename is not None:
		f = open(filename, filemode)

	names = []
	aris = []
	for model in model_list:
		names.append(model.name)
		aris.append(model.ari)

	scores = dict(zip(names, aris))

	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

	print('ARI scores:', file=f)
	print('\033[1m- {0}: {1:.6}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]), file=f)
	for score_tp in sorted_scores[1:]:
		print('- {0}: {1:.6}'.format(score_tp[0], score_tp[1]), file=f)

	if f is not None:
		f.close()

def print_model_nmis(model_list, filename=None, filemode='w'):
	""" Print ordered ARI scores.
	"""
	f = None
	if filename is not None:
		f = open(filename, filemode)

	names = []
	nmis = []
	for model in model_list:
		names.append(model.name)
		nmis.append(model.nmi)

	scores = dict(zip(names, nmis))

	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

	print('NMI scores:', file=f)
	print('\033[1m- {0}: {1:.6}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]), file=f)
	for score_tp in sorted_scores[1:]:
		print('- {0}: {1:.6}'.format(score_tp[0], score_tp[1]), file=f)

	if f is not None:
		f.close()

def print_model_asws(model_list, filename=None, filemode='w'):
	""" Print ordered silhouette scores.
	"""
	f = None
	if filename is not None:
		f = open(filename, filemode)

	names = []
	silhs = []
	for model in model_list:
		names.append(model.name)
		silhs.append(model.silhouette)

	scores = dict(zip(names, silhs))

	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

	print('ASW scores:', file=f)
	print('\033[1m- {0}: {1:.6}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]), file=f)
	for score_tp in sorted_scores[1:]:
		print('- {0}: {1:.6}'.format(score_tp[0], score_tp[1]), file=f)

	if f is not None:
		f.close()


# def print_model_silhouettes(model_list, filename=None, filemode='w'):
# 	""" Print ordered silhouette scores.
# 	"""
# 	f = None
# 	if filename is not None:
# 		f = open(filename, filemode)

# 	names = []
# 	silhs = []
# 	for model in model_list:
# 		names.append(model.name)
# 		silhs.append(model.silhouette)

# 	scores = dict(zip(names, silhs))

# 	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

# 	print('Silhouette scores:', file=f)
# 	print('\033[1m- {0}: {1:.6}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]), file=f)
# 	for score_tp in sorted_scores[1:]:
# 		print('- {0}: {1:.6}'.format(score_tp[0], score_tp[1]), file=f)

# 	if f is not None:
# 		f.close()

def print_model_dropid_accs(model_list, filename=None, filemode='w'):
	f = None
	if filename is not None:
		f = open(filename, filemode)

	names = []
	dropid_accs = []
	for model in model_list:
		names.append(model.name)
		dropid_accs.append(model.dropid_acc)

	scores = dict(zip(names, dropid_accs))

	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

	print('Dropout identification accuracy:', file=f)
	print('\033[1m- {0}: {1:.6}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]), file=f)
	for score_tp in sorted_scores[1:]:
		print('- {0}: {1:.6}'.format(score_tp[0], score_tp[1]), file=f)

	if f is not None:
		f.close()

def print_model_dropimp_errs(model_list, filename=None, filemode='w'):
	f = None
	if filename is not None:
		f = open(filename, filemode)

	names = []
	dropimp_errs= []
	for model in model_list:
		names.append(model.name)
		dropimp_errs.append(model.dropimp_err)

	scores = dict(zip(names, dropimp_errs))

	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=False) # lower is better

	print('Dropout imputation error:', file=f)
	print('\033[1m- {0}: {1:.6}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]), file=f)
	for score_tp in sorted_scores[1:]:
		print('- {0}: {1:.6}'.format(score_tp[0], score_tp[1]), file=f)

	if f is not None:
		f.close()