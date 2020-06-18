"""
Tools to assist in benchmarking ML models and gauging their performance.

Copyright 2018-2020 Toyota Resarch Institute. All rights reserved.
Use of this source code is governed by an Apache 2.0
license that can be found in the LICENSE file.
"""

import numpy as np
from typing import List, Sequence, Hashable
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib as mpl
from matplotlib.patches import Rectangle



def precision_recall(fits: List, labels: List, target)->List[float]:
    """
    Computes the precision and recall and F1 score
    for an individual class label 'target',
    which can be any object with an equivalence relation via ==
    :param fits:
    :param labels:
    :param target:
    :return:
    """
    N = len(labels)

    # Generate the counts of true and false positives
    true_positives = len([True for i in range(N)
                          if (fits[i] == target and labels[i] == target)])
    false_positives = len([True for i in range(N)
                           if (fits[i] == target and labels[i] != target)])
    false_negatives = len([True for i in range(N)
                           if (fits[i] != target and labels[i] == target)])

    if true_positives == 0:
        return [0, 0, 0]

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2.0 * precision * recall / (precision + recall)
    return [precision, recall, f1]


def precision_recall_matrix(fits: List, labels: List, classes: List):
    """
    Computes the precision and recall and F1 score for a set of classes at once

    :param fits:
    :param classes:
    :param labels:
    :return:
    """
    results = []
    for cls in classes:
        results.append(precision_recall(fits, labels, cls))
    return np.array(results)


def confusion_dict(fits, labels, classes: Sequence[Hashable]):
    """
    Generate a confusion matrix in dictionary representation,
    counting the classification error by category.

    :param fits:
    :param labels:
    :param classes:
    :return:
    """
    score = {cls: [0] * len(classes) for cls in classes}
    cls_idxs = {cls: classes.index(cls) for cls in classes}

    for fit, label in zip(fits, labels):
        score[label][cls_idxs[fit]] += 1

    return score


def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_coordination_confusion_matrix(guesses, labels, categories=(4, 5, 6),
                                       title='Confusion Matrix \n  (F1 Score on Diagonal)',
                                       savefig='',
                                       fontcolor='black'):
    """
    Pass in predictions as guesses and labels as labels
    :param save:
    :param fontcolor:
    :param guesses:
    :param labels:
    :param categories:
    :param title:
    :return:
    """

    mpl.rcParams.update({'text.color': fontcolor,
                         'axes.labelcolor': fontcolor})

    conf_dict = confusion_dict(guesses, labels, categories)
    f1_score = precision_recall_matrix(guesses, labels, categories)
    f1s = [np.round(100 * x[2], 1) for x in f1_score]
    conf_mat = np.array([conf_dict[c] for c in categories])
    percent_mat = np.copy(conf_mat).astype('float64')
    fig = plt.figure(1, dpi=300)
    ax = fig.add_subplot(111, autoscale_on=True)
    sums = np.sum(conf_mat, axis=1, dtype='float64')

    for i in range(3):
        percent_mat[i, :] = percent_mat[i, :] / sums[i]

    wrong_mat = np.array([[True, False, False], [False, True, False], [False, False, True]])
    correct_mat = np.array([[False, True, True], [True, False, True], [True, True, False]])

    correct_masked = np.ma.masked_array(percent_mat, correct_mat)
    wrong_masked = np.ma.masked_array(percent_mat, wrong_mat)

    good_norm = plt.Normalize(vmin=min(.5, np.min(correct_masked)), vmax=1)
    bad_norm = plt.Normalize(vmin=0, vmax=max(.5, np.max(wrong_masked)))

    cor_ax = ax.imshow(correct_masked, cmap=cm.get_cmap('Greens'), norm=good_norm,
                       origin='upper')
    wro_ax = ax.imshow(wrong_masked, cmap=cm.get_cmap("Reds"), norm=bad_norm,
                       origin='upper')
    cla = plt.colorbar(cor_ax)
    clb = plt.colorbar(wro_ax)
    cla.set_label("% Class Right")
    clb.set_label("% Class Wrong")

    min_val, max_val = 0, 3
    ind_array = np.arange(min_val + 0.0, max_val + 0.0, 1.0)
    x, y = np.meshgrid(ind_array, ind_array)

    for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):

        value = conf_mat[i // 3][i % 3]
        ax.text(x_val, y_val, value, va='center', ha='center', color='black', size=16)
        if i in [0, 4, 8]:
            ax.text(x_val, y_val + .25, '({})'.format(f1s[i // 3]), va='center', ha='center', color='black', size=14)

    ax.set_yticks([2, 1, 0])

    ax.set_yticklabels(["6-Fold", '5-Fold', '4-Fold'])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["4-Fold", "5-Fold", "6-Fold"])
    ax.set_ylim(2.5, -.5)
    ax.set_xlim(-.5, 2.5)
    ax.set_ylabel("Spectra Labels")
    ax.set_xlabel("Spectra Fitted")
    ax.spines['left'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    ax.tick_params(axis=u'both', which=u'both', length=0)

    ax.set_title(title)

    if savefig == '':
        plt.show()
    else:
        plt.savefig(savefig, dpi=300, transparent=True,format='pdf',
        bbox_inches = 'tight')

_colors_by_pair = {('Ti','O'):'orangered',
                  ('V','O'):'darkorange',
                  ('Cr','O'):'gold',
                  ('Mn','O'):'seagreen',
                  ('Fe','O'):'dodgerblue',
                  ('Co','O'):'navy',
                  ('Ni','O'):'rebeccapurple',
                  ('Cu','O'):"mediumvioletred"}

def plot_parity_plot(y_guesses, y_valid, title,color='blue',
                     savefig=''):


    fig = plt.figure()
    ax = fig.add_subplot(111)
    the_range = y_valid

    bmin = np.min(the_range)
    bmax = np.max(the_range)

    plt.plot([bmin, bmax], [bmin, bmax], color='black', ls='--')
    plt.xlabel("True Bader Charge", size=24)
    plt.ylabel("Predicted\nBader Charge", size=24)
    plt.title("{}O\nRandom Forest Bader Charges".format(title),
              size=24)

    baders = y_valid

    plt.scatter(y_valid, y_guesses,
                color=color, label="Validation Set",
                alpha=.7, s=2)

    mae = np.mean(np.abs(y_valid - y_guesses))
    # mse = np.mean(np.abs(guesses-baders)**2)

    y_val_bar = np.mean(y_valid)
    SStot = np.sum(np.abs(y_valid - y_val_bar) ** 2)
    SSres = np.sum((y_valid - y_guesses ** 2))
    # print(SStot)
    # print(SSres)
    R2 = 1 - SSres / SStot

    extra1 = Rectangle((0, 0), 0, 0, fc='w', fill=False,
                       edgecolor='none', linewidth=0)

    ax.legend([extra1, extra1, extra1], ['MAE = %.3f' % mae],
              fontsize=20, frameon=False, loc='lower right')

    if savefig:
        plt.savefig(savefig,format='pdf',dpi=300,
                    bbox_inches='tight')
    plt.show()



