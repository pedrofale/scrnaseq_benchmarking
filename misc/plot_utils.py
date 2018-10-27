import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import kde
import operator
from collections import OrderedDict

global_palette = sns.color_palette("Set1", 10, desat=.7)

def setBoxColors(bp, colors):
    n_boxes = len(bp['boxes'])
    assert n_boxes <= len(colors)
    
    for i in range(n_boxes):
        # plt.setp(bp['boxes'][i], color=colors[i])
        # plt.setp(bp['caps'][i*2], color=colors[i])
        # plt.setp(bp['caps'][i*2+1], color=colors[i])
        # plt.setp(bp['whiskers'][i*2], color=colors[i])
        # plt.setp(bp['whiskers'][i*2+1], color=colors[i])
        # plt.setp(bp['fliers'][i], color=colors[i])
        plt.setp(bp['medians'][i], color='black')

def plot_simulation_results(low_sep, high_sep, ax=None, legend=None, title=None, ylabel=None,
                            colors = ['blue', 'red', 'green', 'yellow', 'cyan'], show_legend=False):
    # low_sep and high_sep must both be lists of length equal to the number of methods to test (num of boxes)
    # and for each method the array must be of shape nruns

    # if n_boxes is odd, each xticklabel should be at the center box
    # if it is even, each xticklabel should be between the 2 center boxes

    n_boxes = len(low_sep)
    
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    # first boxplot pair
    bp1_pos = range(1, n_boxes + 1)
    bp1 = plt.boxplot(low_sep, positions = bp1_pos, widths = 0.6)
    setBoxColors(bp1, colors)

    # second boxplot pair
    bp2_pos = range(n_boxes + 3, n_boxes + 3 + n_boxes)
    bp2 = plt.boxplot(high_sep, positions = bp2_pos, widths = 0.6)
    setBoxColors(bp2, colors)

    if legend is not None:
        handles = []
        for i in range(n_boxes):
            # draw temporary lines and use them to create a legend
            h, = plt.plot([0,0], colors[i])
            handles.append(h)
        
        if show_legend:
            plt.legend(handles, legend, frameon=True)
            _ = [h.set_visible(False) for h in handles]
    
    # set axes limits and labels
    if show_legend:
        plt.xlim(0,n_boxes*2 + 5)
    else:
        plt.xlim(0, n_boxes*2 + 3)
    ax.set_xticklabels(['Low separability', 'High separability'])
    ax.set_xticks([np.mean(bp1_pos), np.mean(bp2_pos)])

    if ylabel is not None:
        plt.ylabel(ylabel)
    
    if title is not None:
        plt.title(title)

    if not show_legend and legend is not None:
        return ax, handles, legend
    
    return ax    

def plot_boxplot(results_list, names_list, ylabel=None, title=None, filename=None, ax=None):
    # run_results contains a list of 1D arrays with the results to plot. Each element
    # in the list is a box. 

    n_boxes = len(results_list)

    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    # first boxplot pair
    bp1_pos = range(1, n_boxes + 1)
    bp1 = plt.boxplot(results_list, positions = bp1_pos)

    ax.set_xticklabels(names_list)

    if ylabel is not None:
        plt.ylabel(ylabel)
    
    if title is not None:
        plt.title(title)
    
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

def plot_convergence_curves(curve_list, label_list=None, ax=None, legend=None, title='', xlabel='', ylabel='', filename=None):
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    if label_list is not None:
        assert len(curve_list) == len(label_list)
        for i in range(len(curve_list)):
            plt.plot(curve_list[i], label=label_list[i])
    else:
        for i in range(len(curve_list)):
            plt.plot(curve_list[i])
            
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    if legend:
        plt.legend(frameon=True)
    if ax is None:
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()

def plot_model_convergence(model_list, mode='train_ll_time', ax=None, legend=None, title='', xlabel='', ylabel='', filename=None):
    """Takes a list of models as input.
    """
    if mode not in ['train_ll_time', 'train_ll_it', 'test_ll_it', 'silh_time', 'silh_it']:
        return

    if mode == 'train_ll_time':
        curve_list = [model.inf.train_ll_time for model in model_list]
    if mode == 'train_ll_it':
        curve_list = [model.inf.train_ll_it for model in model_list]
    if mode == 'test_ll_it':
        curve_list = [model.inf.test_ll_it for model in model_list]
    if mode == 'silh_time':
        curve_list = [model.inf.silh_time for model in model_list]
    if mode == 'silh_it':
        curve_list = [model.inf.silh_it for model in model_list]

    plot_convergence_curves(curve_list, label_list=[model.name for model in model_list], 
        ax=ax, legend=legend, title=title, xlabel=xlabel, ylabel=ylabel, filename=filename)

def plot_tsne(tsne, clusters, labels=None, ax=None, legend=None, markers=None, title='', xlabel='', ylabel='', s=30, alpha=0.5, bbox_to_anchor=[1., 1.], filename=None, ncol=None):
    if labels is not None:
        labels = np.array(labels)
    
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()
    
    for c in np.unique(clusters):
        if labels is not None:
            ax.scatter(tsne[clusters==c, 0], tsne[clusters==c, 1], s=s, alpha=alpha, label=labels[clusters==c][0])
        else:
            ax.scatter(tsne[clusters==c, 0], tsne[clusters==c, 1], s=s, alpha=alpha)
    
    if labels is not None and legend:
        ax.legend(bbox_to_anchor=bbox_to_anchor, frameon=True, fontsize=14, ncol=ncol)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=14)
    if ax is None:
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()

def plot_2d(model_list, clusters, titles=False, labels=None, ax=None, nrows=1, ncols=None, legend=None, title='', s=30, alpha=0.5, bbox_to_anchor=[1., 1.], filename=None):
    # Sort by decreasing silhouette
    names = []
    tsnes = []
    scores = []
    for model in model_list:
        names.append(model.name)
        tsnes.append(model.proj_2d)
    
    # Plot in decreasing score order
    nresults = len(model_list)
    if ncols is None:
        ncols = nresults

    if ax is None:
        fig = plt.figure(figsize=(16, 4))

    for i in range(nresults):
        ax = plt.subplot(nrows, ncols, i+1)
        plot_tsne(tsnes[i], clusters, labels=labels, ax=ax, s=s, alpha=alpha)
        if titles:
            plt.title(names[i], fontsize=14)
    if labels is not None and legend:
        plt.legend(bbox_to_anchor=bbox_to_anchor, frameon=True, fontsize=14)
    
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_sorted_2d(model_list, clusters, order='silh', titles=False, labels=None, ax=None, nrows=1, ncols=None, legend=None, title='', s=30, alpha=0.5, bbox_to_anchor=[1., 1.], filename=None):
    if order is not None:
        if order not in ['silh', 'ari', 'nmi', 'none']:
            print('{} unrecognized.'.format(order))
            return

    # Sort by decreasing silhouette
    names = []
    tsnes = []
    scores = []
    for model in model_list:
        names.append(model.name)
        tsnes.append(model.proj_2d)
        if order == 'silh':
            scores.append(model.silhouette)
        elif order == 'ari':
            scores.append(model.ari)
        else:
            scores.append(model.nmi)

    if order:
        scores = dict(zip(names, scores))
        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

    else:
        sorted_scores = list(zip(names, scores))
    
    # Plot in decreasing score order
    nresults = len(model_list)
    if ncols is None:
        ncols = nresults

    if ax is None:
        fig = plt.figure(figsize=(16, 4))

    for i in range(nresults):
        ax = plt.subplot(nrows, ncols, i+1)
        plot_tsne(tsnes[names.index(sorted_scores[i][0])], clusters, labels=labels, ax=ax, s=s, alpha=alpha)
        if titles:
            plt.title(sorted_scores[i][0], fontsize)
    if labels is not None and legend:
        plt.legend(bbox_to_anchor=bbox_to_anchor, frameon=True, fontsize=14)
    
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_imputation_density(imputed, true, dropout_idx, title="", ymax=10, nbins=50, ax=None, ylabel=False, cmap="Greys", filename=None, show=True):
    # imputed is NxP 
    # true is NxP
    
    # We only care about the entries affected by dropouts
    y, x = imputed[dropout_idx], true[dropout_idx]
    
    # let's only look at the values that are lower than ymax
    mask = x < ymax
    x = x[mask]
    y = y[mask]
    
    mask = y < ymax
    x = x[mask]
    y = y[mask]
    
    # make the vectors the same size
    l = np.minimum(x.shape[0], y.shape[0])
    x = x[:l]
    y = y[:l]
    
    data = np.vstack([x, y])

    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.gca()

    ax.set_xlim([0, ymax])
    ax.set_ylim([0, ymax])

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(data)
    xi, yi = np.mgrid[0:ymax:nbins*1j, 0:ymax:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=cmap)

    a, _, _, _ = np.linalg.lstsq(x[:,np.newaxis], y, rcond=None)
    l = np.linspace(0, ymax)
    ax.plot(l, a * l, color='black')

    ax.plot(l, l, color='black', linestyle=":")

    plt.title(title, fontsize=12)
    if ylabel:
        plt.ylabel("Imputed counts")
    plt.xlabel("Original counts")

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    elif show:
        plt.show()

def plot_sorted_imputation_densities(model_list, X_train, order=False, ax=None, nrows=1, ncols=None, ymax=10, nbins=50, cmap="Greys", filename=None):
    # Sort by decreasing imputation error
    names = []
    dropimp_errs = []
    est_Rs = []
    for model in model_list:
        names.append(model.name)
        est_Rs.append(model.est_R)
        dropimp_errs.append(model.dropimp_err)

    if order:
        scores = dict(zip(names, dropimp_errs))
        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=False) # lower is better

    else:
        sorted_scores = list(zip(names, dropimp_errs))

    # Plot in decreasing imputation error order
    nresults = len(model_list)
    if ncols is None:
        ncols = nresults

    if ax is None:
        fig = plt.figure(figsize=(20, 4))

    for i in range(nresults):
        ax = plt.subplot(nrows, ncols, i+1)
        plot_imputation_density(est_Rs[names.index(sorted_scores[i][0])], 
            X_train, model_list[i].dropout_idx, ymax=ymax, ax=ax, title=sorted_scores[i][0], ylabel=(i==0), nbins=nbins, cmap=cmap, show=False)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

def get_metrics_lists(model_list):
    """
    example: model_list=[gap1, gap2, gap3]
    returns: {'train_ll': [gap1.train_ll, gap2.train_ll, gap3.train_ll], 'test_ll': [gap1.test_ll, gap2.test_ll, gap3.test_ll], ...}
    """
    train_ll = []
    test_ll = []
    asw = []
    ari = []
    nmi = []

    for model in model_list:
        train_ll.append(model.train_ll)
        test_ll.append(model.test_ll)
        asw.append(model.silhouette)
        ari.append(model.ari)
        nmi.append(model.nmi)

    metrics_lists = {'train_ll': train_ll, 'test_ll': test_ll, 'asw': asw, 'ari': ari, 'nmi': nmi}

    return metrics_lists

def prepare_for_barplots(model_list):
    """
    model_list = [m1, m2, m3]
    returns: {asw: {'m1.name': m1.asw, 'm2.name': m2.asw, 'm3.name': m3.asw},
                ...
                    }
    """
    res = OrderedDict()
    res['train_ll'] = {model.name: model.train_ll for model in model_list}
    res['test_ll'] = {model.name: model.test_ll for model in model_list}
    res['asw'] = {model.name: model.asw for model in model_list}
    res['ari'] = {model.name: model.ari for model in model_list}
    res['nmi'] = {model.name: model.nmi for model in model_list}
    try:
        res['basw'] = {model.name: model.batch_asw for model in model_list}
    except:
        pass

    return res

def prepare_for_boxplots(model_list_cv):
    """
    model_list_cv = [[m1, m1, m1], [m2, m2, m2]]

    return: {'asw': {'m1.name': [m1.asw, m1.asw, m1.asw], 'm2.name': [m2.asw, m2.asw, m2.asw]},
             'nmi': {'m1.name': [m1.nmi, m1.nmi, m1.nmi], 'm2.name': [m2.nmi, m2.nmi, m2.nmi]},
             ...
                }
    """
    res = OrderedDict()
    res['train_ll'] = {model_list[0].name: [model.train_ll for model in model_list] for model_list in model_list_cv}
    res['test_ll'] = {model_list[0].name: [model.test_ll for model in model_list] for model_list in model_list_cv}
    res['asw'] = {model_list[0].name: [model.asw for model in model_list] for model_list in model_list_cv}
    res['ari'] = {model_list[0].name: [model.ari for model in model_list] for model_list in model_list_cv}
    res['nmi'] = {model_list[0].name: [model.nmi for model in model_list] for model_list in model_list_cv}
    try:
        res['basw'] = {model_list[0].name: [model.batch_asw for model in model_list] for model_list in model_list_cv}
    except:
        pass
    try:
        res['dropimp_err'] = {model_list[0].name: [model.dropimp_err for model in model_list] for model_list in model_list_cv}
    except:
        pass
    try:
        res['corr_1_ls'] = {model_list[0].name: [model.corr_1_ls for model in model_list] for model_list in model_list_cv}
        res['corr_2_ls'] = {model_list[0].name: [model.corr_2_ls for model in model_list] for model_list in model_list_cv}

        res['corr_1_dr'] = {model_list[0].name: [model.corr_1_dr for model in model_list] for model_list in model_list_cv}
        res['corr_2_dr'] = {model_list[0].name: [model.corr_2_dr for model in model_list] for model_list in model_list_cv}
    except:
        pass

    return res

def clustering_barplot(model_list, ax=None, title=None, ylabel=None,
                            colors = ['blue', 'orange', 'green', 'red', 'yellow', 'purple', 'gray'], do_legend=True, show_legend=False, ylim=None):


    colors = global_palette
    res = prepare_for_barplots(model_list)
    try:
        basw = res['basw']
        if basw[0][0] == None:
            basw = None
    except:
        basw = None

    try:
        corr_1_ls = res['corr_1_ls']
        corr_2_ls = res['corr_2_ls']
        corr_1_dr = res['corr_1_dr']
        corr_2_dr = res['corr_2_dr']
        if corr_1_ls[0][0] == None:
            corr_1_ls = None
            corr_2_ls = None
            corr_1_dr = None
            corr_2_dr = None
    except:
        corr_1_ls = None
        corr_2_ls = None
        corr_1_dr = None
        corr_2_dr = None

    return _clustering_barplot(res['asw'], res['ari'], res['nmi'], basw, corrs=[corr_1_ls, corr_2_ls, corr_1_dr, corr_2_dr], 
        ax=ax, title=title, ylabel=ylabel, colors=colors, do_legend=do_legend, show_legend=show_legend, ylim=ylim)

def loglikelihood_barplot(model_list, ax=None, title=None, ylabel=None,
                            colors = ['blue', 'orange', 'green', 'red', 'yellow', 'purple', 'gray'], do_legend=True, show_legend=False, ylim=None, hatches=None):

    colors = global_palette
    res = prepare_for_barplots(model_list)
    return _loglikelihood_barplot(res['train_ll'], res['test_ll'], ax=ax, title=title, ylabel=ylabel, colors=colors, do_legend=do_legend, show_legend=show_legend, 
        ylim=ylim, hatches=hatches)


def _clustering_barplot(asw, ari, nmi, basw=None, corrs=None, ax=None, title=None, ylabel=None,
                            colors = ['blue', 'orange', 'green', 'red', 'yellow', 'purple', 'gray'], do_legend=True, show_legend=False, ylim=None):
    """
    asw, ari and nmi are, each one, a list containing the scores for each model across all runs.
    example: asw = {'model1':model1_score, 'model2': model2_score, 'model3': model3_score}
    """
    legend = asw.keys()
    asw = list(asw.values())
    ari = list(ari.values())
    nmi = list(nmi.values())
    metrics = [asw, ari, nmi]

    if basw is not None:
        basw = list(basw.values())
        if basw[0] is not None:
            metrics = [asw, ari, nmi, basw]

    n_boxes = len(asw)
    
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    init = 1
    ticks_pos = []
    for i in range(len(metrics)):
        stop = init + n_boxes
        bp_pos = range(init, stop)
        ticks_pos.append(np.mean(bp_pos))

        bp = plt.bar(bp_pos, metrics[i], color=colors)
        
        init = stop + 2

    if do_legend is True:
        handles = []
        for i in range(n_boxes):
            # draw temporary lines and use them to create a legend
            h, = plt.plot([0,0], color=colors[i])
            handles.append(h)

        if show_legend:
            plt.legend(handles, legend, frameon=True, bbox_to_anchor=[1., 1.])
        _ = [h.set_visible(False) for h in handles]
    
    # set axes limits and labels
    if show_legend:
        plt.xlim(0, stop)
    else:
        plt.xlim(0, stop)
    if basw is not None:
        ax.set_xticklabels(['ASW', 'ARI', 'NMI', 'bASW'])
    else:
        ax.set_xticklabels(['ASW', 'ARI', 'NMI', 'bASW'])
    ax.set_xticks(ticks_pos)

    if ylim is not None:
        plt.ylim(ylim)

    if ylabel is not None:
        plt.ylabel(ylabel)
    
    if title is not None:
        plt.title(title)

    if not show_legend and do_legend:
        return ax, handles, legend
    
    return ax


def _loglikelihood_barplot(train_ll, test_ll, ax=None, title=None, ylabel=None,
                            colors = ['blue', 'orange', 'green', 'red', 'yellow', 'purple', 'gray'], do_legend=True, show_legend=False, ylim=None, hatches=None):
    """
    asw, ari and nmi are, each one, a list containing the scores for each model across all runs.
    example: asw = {'model1':model1_score, 'model2': model2_score, 'model3': model3_score}
    """
    legend = train_ll.keys()
    train_ll = list(train_ll.values())
    test_ll = list(test_ll.values())
    metrics = [train_ll, test_ll]
    ticks_pos = []

    n_boxes = len(train_ll)
    
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    init = 1
    for i in range(2):
        stop = init + n_boxes
        bp_pos = range(init, stop)
        ticks_pos.append(np.mean(bp_pos))

        bp = plt.bar(bp_pos, metrics[i], color=colors)
        if hatches is not None:
            for i, thisbar in enumerate(bp.patches):
                # Set a different hatch for each bar
                thisbar.set_hatch(hatches[i])
        
        init = stop + 2

    if do_legend is True:
        handles = []
        for i in range(n_boxes):
            # draw temporary bars and use them to create a legend
            if hatches is not None:
                h, = plt.bar(0., 0., color=colors[i], hatch=hatches[i])
            else:
                h, = plt.bar(0., 0., color=colors[i])
            handles.append(h)

        if show_legend:
            plt.legend(handles, legend, frameon=True, bbox_to_anchor=[1., 1.], fontsize=14)
        _ = [h.set_visible(False) for h in handles]
    
    # set axes limits and labels
    if show_legend:
        plt.xlim(0, stop)
    else:
        plt.xlim(0, stop)
    ax.set_xticklabels(['Train LL', 'Test LL'], fontsize=14)
    ax.set_xticks(ticks_pos)

    if ylim is not None:
        plt.ylim(ylim)

    if ylabel is not None:
        plt.ylabel(ylabel)
    
    if title is not None:
        plt.title(title)

    if not show_legend and do_legend:
        return ax, handles, legend
    
    return ax
    
def clustering_cv(model_list_cv, ax=None, title=None, ylabel=None,
                            colors=None, do_legend=True, show_legend=False, ylim=None, box=False, hatches=None):

    
    if colors is None:
        colors = global_palette
    res = prepare_for_boxplots(model_list_cv)
    try:
        basw = res['basw']
        if list(basw.values())[0][0] == None:
            basw = None
    except:
        basw = None

    try:
        corr_1_ls = res['corr_1_ls']
        corr_2_ls = res['corr_2_ls']
        corr_1_dr = res['corr_1_dr']
        corr_2_dr = res['corr_2_dr']
        if list(corr_1_ls.values())[0][0] == None:
            corr_1_ls = None
            corr_2_ls = None
            corr_1_dr = None
            corr_2_dr = None
    except:
        corr_1_ls = None
        corr_2_ls = None
        corr_1_dr = None
        corr_2_dr = None

    return _clustering_cv(res['asw'], res['ari'], res['nmi'], basw, corrs=[corr_1_ls, corr_2_ls, corr_1_dr, corr_2_dr], ax=ax, title=title, ylabel=ylabel, colors=colors, do_legend=do_legend, 
        show_legend=show_legend, ylim=ylim, box=box, hatches=hatches)

def loglikelihood_cv(model_list_cv, ax=None, title=None, ylabel=None,
                            colors = None, do_legend=True, show_legend=False, ylim=None, box=False, hatches=None):

    if colors is None:
        colors = global_palette
    res = prepare_for_boxplots(model_list_cv)
    return _loglikelihood_cv(res['train_ll'], res['test_ll'], ax=ax, title=title, ylabel=ylabel, colors=colors, do_legend=do_legend, show_legend=show_legend, ylim=ylim, box=box, hatches=hatches)


def imputationerr_boxplot(model_list, ax=None, title=None, ylabel=None,
                            colors = None, do_legend=True, show_legend=False, ylim=None, box=False, hatches=None):
    
    if colors is None:
        colors = global_palette
    res = prepare_for_boxplots(model_list)
    return _dropimperr_boxplot(res['dropimp_err'], ax=ax, title=title, ylabel=ylabel, colors=colors, do_legend=do_legend, show_legend=show_legend, ylim=ylim, box=box, hatches=hatches)


def _clustering_cv(asw, ari, nmi, basw=None, corrs=None, ax=None, title=None, ylabel=None,
                            colors = ['blue', 'red', 'green', 'orange', 'yellow'], do_legend=True, show_legend=False, ylim=None, box=False, hatches=None):
    """
    asw, ari and nmi are, each one, a list containing the scores for each model across all runs.
    example: asw = {'model1':model1_scores, 'model2': model2_scores, 'model3': model3_scores}, where model#_scores=[run1, run2, run3, run4, run5]
    """
    legend = asw.keys()
    asw = list(asw.values())
    ari = list(ari.values())
    nmi = list(nmi.values())
    metrics = [asw, ari, nmi]
    if basw is not None:
        basw = list(basw.values())
        if basw[0] is not None:
            metrics = [asw, ari, nmi, basw]

    if corrs is not None:
        corr_1_ls = corrs[0]
        corr_2_ls = corrs[1]
        corr_1_dr = corrs[2]
        corr_2_dr = corrs[3]
        
        corr_1_ls = list(corr_1_ls.values())
        corr_2_ls = list(corr_2_ls.values())
        corr_1_dr = list(corr_1_dr.values())
        corr_2_dr = list(corr_2_dr.values())
        if corr_1_ls[0] is not None:
            metrics = [asw, corr_1_ls, corr_2_ls, corr_1_dr, corr_2_dr]

    ticks_pos = []

    n_boxes = len(asw)

    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    if box:
        init = 1
        for i in range(len(metrics)):
            stop = init + n_boxes
            bp_pos = range(init, stop)
            ticks_pos.append(np.mean(bp_pos))

            bp = plt.boxplot(metrics[i], positions=bp_pos, widths=0.6, patch_artist=True)
            setBoxColors(bp, colors)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            init = stop + 2
    else:
        init = 1
        for i in range(len(metrics)):
            stop = init + n_boxes
            bp_pos = range(init, stop)
            ticks_pos.append(np.mean(bp_pos))

            means = []
            stds = []
            for score_list in metrics[i]:
                means.append(np.mean(score_list))
                stds.append(np.std(score_list))

            bp = plt.bar(bp_pos, means, yerr=stds, color=colors)
            if hatches is not None:
                for i, thisbar in enumerate(bp.patches):
                    # Set a different hatch for each bar
                    thisbar.set_hatch(hatches[i])

            # plt.bar(bp_pos, means, color=colors)
            # bp = plt.boxplot(metrics[i], positions=bp_pos, widths=0.6, patch_artist=True)
            # setBoxColors(bp, colors)

            init = stop + 2

    if do_legend is True:
        handles = []
        for i in range(n_boxes):
            # draw temporary lines and use them to create a legend
            if hatches is not None:
                h, = plt.bar(0., 0., color=colors[i], hatch=hatches[i])
            else:
                h, = plt.bar(0., 0., color=colors[i])
            handles.append(h)

        if show_legend:
            plt.legend(handles, legend, frameon=True, bbox_to_anchor=[1., 1.], fontsize=14)
        _ = [h.set_visible(False) for h in handles]
    
    # set axes limits and labels
    if show_legend:
        plt.xlim(0, stop)
    else:
        plt.xlim(0, stop)

    if corrs is not None:
        ax.set_xticklabels(['ASW', 'Corr1_LS', 'Corr2_LS', 'Corr1_DR', 'Corr2_DR'], fontsize=14)
    else:
        if basw is not None:
            ax.set_xticklabels(['ASW', 'ARI', 'NMI', 'bASW'], fontsize=14)
        else:
            ax.set_xticklabels(['ASW', 'ARI', 'NMI'], fontsize=14)
    ax.set_xticks(ticks_pos)

    if ylim is not None:
        plt.ylim(ylim)

    if ylabel is not None:
        plt.ylabel(ylabel)
    
    if title is not None:
        plt.title(title)

    if not show_legend and do_legend:
        return ax, handles, legend
    
    return ax

def _loglikelihood_cv(train_ll, test_ll, ax=None, title=None, ylabel=None,
                            colors = ['blue', 'red', 'green', 'orange', 'yellow'], do_legend=True, show_legend=False, ylim=None, box=False, hatches=None):
    """
    asw, ari and nmi are, each one, a list containing the scores for each model across all runs.
    example: asw = {'model1':model1_scores, 'model2': model2_scores, 'model3': model3_scores}, where model#_scores=[run1, run2, run3, run4, run5]
    """
    legend = train_ll.keys()
    train_ll = list(train_ll.values())
    test_ll = list(test_ll.values())
    metrics = [train_ll, test_ll]
    ticks_pos = []

    n_boxes = len(train_ll)
    
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    if box:
        init = 1
        for i in range(2):
            stop = init + n_boxes
            bp_pos = range(init, stop)
            ticks_pos.append(np.mean(bp_pos))

            bp = plt.boxplot(metrics[i], positions=bp_pos, widths=0.6, patch_artist=True)
            setBoxColors(bp, colors)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            init = stop + 2

    else:
        init = 1
        for i in range(len(metrics)):
            stop = init + n_boxes
            bp_pos = range(init, stop)
            ticks_pos.append(np.mean(bp_pos))

            means = []
            stds = []
            for score_list in metrics[i]:
                means.append(np.mean(score_list))
                stds.append(np.std(score_list))

            bp = plt.bar(bp_pos, means, yerr=stds, color=colors)
            if hatches is not None:
                for i, thisbar in enumerate(bp.patches):
                    # Set a different hatch for each bar
                    thisbar.set_hatch(hatches[i])

            # plt.bar(bp_pos, means, color=colors)
            # bp = plt.boxplot(metrics[i], positions=bp_pos, widths=0.6, patch_artist=True)
            # setBoxColors(bp, colors)

            init = stop + 2

    if do_legend is True:
        handles = []
        for i in range(n_boxes):
            # draw temporary lines and use them to create a legend
            if hatches is not None:
                h, = plt.bar(0., 0., color=colors[i], hatch=hatches[i])
            else:
                h, = plt.bar(0., 0., color=colors[i])
            handles.append(h)

        if show_legend:
            plt.legend(handles, legend, frameon=True, bbox_to_anchor=[1., 1.], fontsize=14)
            _ = [h.set_visible(False) for h in handles]
    
    # set axes limits and labels
    if show_legend:
        plt.xlim(0, stop)
    else:
        plt.xlim(0, stop)
    ax.set_xticklabels(['Train LL', 'Test LL'], fontsize=14)
    ax.set_xticks(ticks_pos)

    if ylim is not None:
        plt.ylim(ylim)

    if ylabel is not None:
        plt.ylabel(ylabel)
    
    if title is not None:
        plt.title(title)

    if not show_legend and do_legend:
        return ax, handles, legend
    
    return ax

def _dropimperr_boxplot(dropimp_err, ax=None, title=None, ylabel=None,
                            colors = ['blue', 'red', 'green', 'orange', 'yellow'], do_legend=True, show_legend=False, ylim=None, hatches=None, box=None):
    """
    dropimp_err is a list containing the dropout imputation error for each model across all runs.
    example: dropimp_err = {'model1':model1_scores, 'model2': model2_scores, 'model3': model3_scores}, where model#_scores=[run1, run2, run3, run4, run5]
    """
    legend = dropimp_err.keys()
    ticks_pos = []
    dropimp_err = list(dropimp_err.values())
    metrics = [dropimp_err]
    n_boxes = len(dropimp_err)
    
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    # init = 1
    # for i in range(1):
    #     stop = init + n_boxes
    #     bp_pos = range(init, stop)
    #     ticks_pos.append(np.mean(bp_pos))

    #     bp = plt.boxplot(metrics[i], positions=bp_pos, widths=0.6, patch_artist=True)
    #     # setBoxColors(bp, colors)
        
    #     init = stop + 2

    #     for patch, color in zip(bp['boxes'], colors):
    #         patch.set_facecolor(color)

    if box:
        init = 1
        for i in range(len(metrics)):
            stop = init + n_boxes
            bp_pos = range(init, stop)
            ticks_pos.append(np.mean(bp_pos))

            bp = plt.boxplot(metrics[i], positions=bp_pos, widths=0.6, patch_artist=True)
            setBoxColors(bp, colors)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            init = stop + 2

    else:
        init = 1
        for i in range(len(metrics)):
            stop = init + n_boxes
            bp_pos = range(init, stop)
            ticks_pos.append(np.mean(bp_pos))

            means = []
            stds = []
            for score_list in metrics[i]:
                means.append(np.mean(score_list))
                stds.append(np.std(score_list))

            bp = plt.bar(bp_pos, means, yerr=stds, color=colors)
            if hatches is not None:
                for i, thisbar in enumerate(bp.patches):
                    # Set a different hatch for each bar
                    thisbar.set_hatch(hatches[i])

            # plt.bar(bp_pos, means, color=colors)
            # bp = plt.boxplot(metrics[i], positions=bp_pos, widths=0.6, patch_artist=True)
            # setBoxColors(bp, colors)

            init = stop + 2

    if do_legend is True:
        handles = []
        for i in range(n_boxes):
            # draw temporary lines and use them to create a legend
            if hatches is not None:
                h, = plt.bar(0., 0., color=colors[i], hatch=hatches[i])
            else:
                h, = plt.bar(0., 0., color=colors[i])
            handles.append(h)

        if show_legend:
            plt.legend(handles, legend, frameon=True, bbox_to_anchor=[1., 1.], fontsize=14)
        _ = [h.set_visible(False) for h in handles]
    
    ax.set_xticks([])

    # set axes limits and labels
    if show_legend:
        plt.xlim(0, stop)
    else:
        plt.xlim(0, stop)

    if ylim is not None:
        plt.ylim(ylim)

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=14)
    
    if title is not None:
        plt.title(title)

    if not show_legend and do_legend:
        return ax, handles, legend
    
    return ax