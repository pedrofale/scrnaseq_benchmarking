"""
Helper functions for training scVI
"""

import time 
import numpy as np
from state_of_art.scVI.benchmarking import * 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def train_model(model, expression_train, sess, num_epochs=250, expression_test=None, step=None, batch_train=None, batch_test=None, kl=None, batch_size=128, max_time=60):
    scVI_batch = batch_train is not None
    
    if step is None:
        step = model.train_step
    
    if batch_size > expression_train.shape[0]:
        batch_size = expression_train.shape[0]

    iterep = int(expression_train.shape[0]/float(batch_size))
    total_n_iter = max(iterep * num_epochs, 1)
    
    training_history = {"t_loss":[], "v_loss":[], "time":[], "epoch":[]}
    training_history["n_hidden"] = model.n_hidden
    training_history["model"] = model.__class__.__name__
    training_history["n_input"] = model.n_input
    training_history["dropout_nn"] = model.dropout_rate
    training_history["dispersion"] = model.dispersion
    training_history["n_layers"] = model.n_layers
    if kl is None:
        warmup = lambda x: np.minimum(1, x / 400.)
    else:
        warmup = lambda x: kl
    
    begin = time.time()    
    for t in range(total_n_iter):
        # warmup
        end_epoch, epoch = t % iterep == 0, t / iterep
        kl = warmup(epoch)
    
        # arange data in batches
        index_train = np.random.choice(np.arange(expression_train.shape[0]), size=batch_size)
        x_train = expression_train[index_train].astype(np.float32)

        #prepare data dictionaries
        dic_train = {model.expression: x_train, model.training_phase:True, model.kl_scale:kl}
        if expression_test is not None:
            dic_test = {model.expression: expression_test,  model.training_phase:False, model.kl_scale:kl}
        
        if scVI_batch:
            b_train = batch_train[index_train]
            dic_train[model.batch_ind] = b_train
            #dic_train[model.mmd_scale] = 10
            if expression_test is not None:
                dic_test[model.batch_ind] = batch_test
                #dic_test[model.mmd_scale] = 10

        
        # run an optimization set
        _, l_tr = sess.run([model.train_step, model.loss], feed_dict=dic_train)

        if end_epoch:          
            
            now = time.time()
            elapsed = now - begin
            m, s = divmod(elapsed, 60)
            h, m = divmod(m, 60)

            if expression_test is not None:
                l_t = sess.run((model.loss), feed_dict=dic_test)

            training_history["t_loss"].append(l_tr)
            if expression_test is not None:
                training_history["v_loss"].append(l_t)
            training_history["time"].append(format_time(int(now-begin)))
            training_history["epoch"].append(epoch)
            
            print("Iteration {0}/{1}. Train loss: {2:.3f}. Elapsed: {3:.0f}h{4:.0f}m{5:.0f}s".format(t+1, total_n_iter, l_tr, h, m, s), end="\r")

            if elapsed > max_time:
                break

            if np.isnan(l_tr):
                break
        
    return training_history


def eval_params(model, data, sess, batch=None):
    dic_full = {model.expression: data, model.training_phase:False, model.kl_scale:1}
    if batch is not None:
        dic_full[model.batch_ind] = batch
        dic_full[model.mmd_scale] = 0 
    rate, dropout = sess.run((model.px_rate, model.px_dropout), feed_dict=dic_full)
    dispersion = np.tile(sess.run((tf.exp(model.px_r))), (rate.shape[0], 1))
    return rate, dispersion, dropout


def eval_imputed_data(model, corrupted_info, expression_train, sess, batch=None): 
    
    (X_zero, i0, j0, ix0) = corrupted_info
    dic_zero = {model.expression: X_zero, model.training_phase:False, model.kl_scale:1.} 
    if batch is not None:
        dic_zero[model.batch_ind] = batch[:, 0]
        dic_zero[model.mmd_scale] = 0
        
    rate_  = sess.run((model.px_rate), \
                                       feed_dict=dic_zero)
    return imputation_error(rate_, expression_train, X_zero, i0, j0, ix0)

def eval_likelihood(model, data, sess, batch=None):  
    dic_full = {model.expression: data, model.training_phase:False, model.kl_scale:1}
    if batch is not None:
        dic_full[model.batch_ind] = batch[:, 0]
        #dic_full[model.mmd_scale] = 0  
    return - sess.run(model.loss, feed_dict=dic_full)  

def eval_latent(model, data, sess, batch=None):
    dic_full = {model.expression: data, model.training_phase:False, model.kl_scale:1}
    if batch is not None:
        dic_full[model.batch_ind] = batch[:, 0]
        dic_full[model.mmd_scale] = 0 
    return sess.run(model.z, feed_dict=dic_full)

def plot_training_info(result):
    plt.plot(result["epoch"], result["t_loss"])
    plt.plot(result["epoch"], result["v_loss"])
    plt.xlabel("number of epochs")
    plt.ylabel("objective function")
    plt.tight_layout()

def show_tSNE(latent, labels, cmap=plt.get_cmap("tab10", 7), return_tSNE=False):
    
    if latent.shape[1] != 2:
        latent = TSNE().fit_transform(latent)
        
    plt.figure(figsize=(10, 10))
    plt.scatter(latent[:, 0], latent[:, 1], c=labels, \
                                   cmap=cmap, edgecolors='none')
    plt.axis("off")
    plt.tight_layout()
    
    if return_tSNE:
        return latent