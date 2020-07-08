# Copyright 2019-2020 Toyota Research Institute. All rights reserved.

from tqdm import tqdm, tqdm_notebook
from sklearn.feature_selection import SelectFromModel
from pprint import pprint
import json
import os
import numpy as np
import sklearn
import pandas as pd
import numpy as np
rseed = 42
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.cm as cm
import matplotlib.colors as colors
import collections
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.collections import LineCollection
import scipy.special as spec
from scipy.stats import norm

import matplotlib.patches as patches


def label_to_color_codex(label):
    """
    Takes in a label for a fitting polynomial
    :param label:
    :return:
    """
    fingerprint = label.split(',')
    fingerprint = [x.split(':') for x in fingerprint]
    fingerprint = {x[0] :x[1] for x in fingerprint}
    for key in ['deg' ,'fraction_size' ,'chunk' ,'coef']:
        fingerprint[key] = int(fingerprint[key])

    idx = fingerprint['chunk']
    if fingerprint['loc' ] =='post':
        idx += fingerprint['fraction_size']
    return fingerprint['fraction_size'] ,idx


def label_to_hr_and_color(label, pre_post=True,
                          fraction=True):

    if label =='peak':
        return 'Peak Location', 'black'
    elif label =='random':
        return 'Random', 'black'
    else:

        fingerprint = label.split(',')

        fingerprint = [x.split(':') for x in fingerprint]
        fingerprint = {x[0]: x[1] for x in fingerprint}
        for key in ['deg', 'fraction_size', 'chunk', 'coef']:
            fingerprint[key] = int(fingerprint[key])

        # print(fingerprint)
        coeff = fingerprint['coef']

    color_list = plt.cm.tab20(np.arange(0, 20))
    color_idx = 0

    hrlabel = ''

    tab10_colors = ['tab:blue', 'tab:orange',
                    'tab:green', 'tab:red',
                    'tab:purple', 'tab:brown',
                    'tab:pink', 'tab:gray',
                    'tab:olive', 'tab:cyan']

    if fingerprint['fraction_size'] == 1:
        if fingerprint['loc'] == 'pre':
            hrlabel += 'Entire Pre-Edge- '
            color = 'tab:blue'
        elif fingerprint['loc'] == 'post':
            hrlabel += 'Entire Post-Edge- '
            color = 'tab:orange'


    elif fingerprint['fraction_size'] == 2:
        hrlabel += 'Twofold- '
        if fingerprint['loc'] == 'pre':
            if fingerprint['chunk'] == 0:
                color = 'tab:blue'
            else:
                color = 'tab:orange'
        if fingerprint['loc'] == 'post':
            if fingerprint['chunk'] == 0:
                color = 'tab:green'
            else:
                color = 'tab:red'


    elif fingerprint['fraction_size'] == 4:
        hrlabel += 'Fourfold- '

        if fingerprint['loc'] == 'post':
            color_idx += 4
        color = tab10_colors[color_idx + fingerprint['chunk']]

    elif fingerprint['fraction_size'] == 5:
        hrlabel += 'Fivefold- '
        if fingerprint['loc'] == 'post':
            color_idx += 5
        color = tab10_colors[color_idx + fingerprint['chunk']]

    elif fingerprint['fraction_size'] == 10:
        hrlabel += 'Tenfold-'
        if fingerprint['loc'] == 'post':
            color_idx += 10

        color = tab20(color_idx + fingerprint['chunk'])

    if coeff == 0:
        hrlabel += 'Const.'
    elif coeff == 1:
        hrlabel += 'Linear'
    elif coeff == 2:
        hrlabel += 'Quad.'
    elif coeff == 3:
        hrlabel += 'Cubic'
    return hrlabel, color



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap



color_ranking = ['red','lightcoral','salmon','orange','darkorange',
                'yellow','gold','olive','green','mediumseagreen','teal','blue','navy',
                'slateblue','darkviolet','purple','violet','orchid']
split_to_title ={1:"Entire Region",2:'Twofold',4:'Fourfold',
                 5:'Fivefold',10:'Tenfold'}



tab02 = plt.get_cmap('tab10')
left_cmap_2 = truncate_colormap(tab02,minval=0,maxval=.2,n=100)
right_cmap_2 = truncate_colormap(tab02,minval=.2,maxval=.4,n=100)

tab04 = plt.get_cmap('tab10')
left_cmap_4 = truncate_colormap(tab04,minval=0,maxval=.2,n=100)
right_cmap_4 = truncate_colormap(tab04,minval=.2,maxval=.4,n=100)

tab08 = plt.get_cmap('tab10')
left_cmap_8 = truncate_colormap(tab08,minval=0,maxval=.4,n=100)
right_cmap_8 = truncate_colormap(tab08,minval=.4,maxval=.8,n=100)

tab10 = plt.get_cmap('tab10')
left_cmap_10 = truncate_colormap(tab10,minval=0,maxval=.5,n=100)
right_cmap_10 = truncate_colormap(tab10,minval=.5,maxval=1.0,n=100)

tab20 = plt.get_cmap('tab20')
left_cmap_20 = truncate_colormap(tab20,minval=0,maxval=.5,n=100)
right_cmap_20 = truncate_colormap(tab20,minval=.5,maxval=1.0,n=100)

fake_X = np.linspace(-5,5,500)
fake_Y = spec.expit(fake_X) + 2*norm.pdf(fake_X,-.7,.8)+ .3*norm.pdf(fake_X,
                                                                 -3.2,.4)
pi = np.argmax(fake_Y)

_cmap_lookup = {2:  (left_cmap_2, right_cmap_2),
    4:  (left_cmap_4, right_cmap_4),
    8:  (left_cmap_8, right_cmap_8),
    10: (left_cmap_10, right_cmap_10),
    20: (left_cmap_20, right_cmap_20)}

fine_splits = [1,2,4,5,10]



def _domain_to_label(domain):

    range = domain[-1]-domain[0]

    dx = range/5

    steps = [domain[0], domain[0]+dx, domain[0]+2*dx,domain[0]+3*dx,
        domain[0] + 4*dx, domain[0]+5*dx]

    labels = [int(np.round(step, 0)) for step in steps]

    return labels




def poly_fit_feature_rank(forest, sorted_keys, title='', x_domain = None,
                          savefig=''):
    """
    Produce a feature ranking diagram for the polynomial fitted
    random forests.

    :param forest:
    :param sorted_keys:
    :param title:
    :return:
    """

    importances = list(forest.feature_importances_)

    # Match feature importances with names of features, the names of which
    # are passed in via sorted_keys
    feature_importances = []
    for i, key in enumerate(sorted_keys):
        feature_importances.append((key, importances[i]))
    feature_importances = sorted(feature_importances, key=lambda x: x[1],
                                 reverse=True)
    mean_imp = np.mean(importances)
    max_imp = np.max(importances)

    # Rank the top importances by taking the top 15 above the mean
    top_importances = [(x[0], x[1] / max_imp) for x in feature_importances if
                       x[1] > mean_imp][:15]
    if len(top_importances)<14:
        print("Warning- top importances not that good")
    labels = [ti[0] for ti in top_importances]
    # Generate the x and y bar plot parts
    bar_x = np.arange(len(top_importances))
    bar_y = [ti[1] for ti in top_importances]

    # Generate the tuples which will index the 'color codex' dictionary,
    # which will then be accessed to color in the sample spectra
    color_keys = [(1, 0), (1, 1)]
    for i in range(4): color_keys.append((2, i))
    for i in range(8): color_keys.append((4, i))
    for i in range(10): color_keys.append((5, i))
    for i in range(20): color_keys.append((10, i))
    color_codex = {color_keys[i]: 'silver' if i % 2 == 0 else 'silver' for i in
                   range(len(color_keys))}
    color_codex['peak'] = 'black'

    # Loop through the labels  and set the color codex to run with the
    # rankings
    for i, l in enumerate(labels):
        if l =='peak':
            color_codex['peak'] = color_ranking[i]
            continue
        elif l.lower() =='random':
            continue

        cc_index = label_to_color_codex(l)
        if color_codex[cc_index] == 'gray' \
                or color_codex[cc_index] == 'silver':
            color_codex[label_to_color_codex(l)] = color_ranking[i]

    fig3 = plt.figure(figsize=(12, 6), constrained_layout=True, dpi=300)
    gs = fig3.add_gridspec(3, 4)

    # Generate the full pleasantville subplot
    f3_ax0 = fig3.add_subplot(gs[:2, :2])

    f3_ax1 = fig3.add_subplot(gs[0, 2])
    f3_ax2 = fig3.add_subplot(gs[0, 3])
    f3_ax3 = fig3.add_subplot(gs[1, 2])
    f3_ax4 = fig3.add_subplot(gs[1, 3])
    f3_ax5 = fig3.add_subplot(gs[2, 2:])
    all_axes = [f3_ax0, f3_ax1, f3_ax2, f3_ax3, f3_ax4, f3_ax5]

    for ax in all_axes:
        ax.spines['left'].set_linewidth(3)
        ax.spines['top'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)

    ideal_axes = [f3_ax1, f3_ax2, f3_ax3, f3_ax4, f3_ax5]

    # loop through each axis and put in the fake spectra colore appropriately
    for ax, split_num in zip(ideal_axes, fine_splits):

        lw = 5
        X_l = fake_X[:pi]
        X_r = fake_X[pi:]

        Y_l = fake_Y[:pi]
        Y_r = fake_Y[pi:]

        if split_num == 1:
            ax.plot(X_l, Y_l, color=color_codex[(1, 0)], lw=lw)
            ax.plot(X_r, Y_r, color=color_codex[(1, 1)], lw=lw)

        else:
            step = len(X_l) // split_num
            split_idx_l = [step * i for i in range(split_num)]
            split_idx_l.append(len(X_l))

            step = len(X_r) // split_num
            split_idx_r = [step * i for i in range(split_num)]
            split_idx_r.append(len(X_r))

            for i in range(split_num):
                sub_x = X_l[split_idx_l[i]:split_idx_l[i + 1]]
                sub_y = Y_l[split_idx_l[i]:split_idx_l[i + 1]]

                ax.plot(sub_x, sub_y, color=color_codex[(split_num, i)], lw=lw)

            for i in range(split_num):
                sub_x = X_r[split_idx_r[i]:split_idx_r[i + 1]]
                sub_y = Y_r[split_idx_r[i]:split_idx_r[i + 1]]

                ax.plot(sub_x, sub_y,
                        color=color_codex[(split_num, i + split_num)], lw=lw)

        ax.set_xlim(-5, 5)
        ax.set_ylim(-.1, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axvline(fake_X[pi], color=color_codex['peak'], ls='--', lw=5)
        ax.set_title(split_to_title[split_num], color='black', size=13)

    if x_domain is not None:
        bottom_ticks = [-5, -5+2, -5+4, -5+6, -5+8, -5+10]
        f3_ax5.set_xticks(ticks=bottom_ticks)
        f3_ax5.set_xticklabels(_domain_to_label(x_domain))
        f3_ax5.set_xlabel('Energy (eV)')

    hr_labels = [label_to_hr_and_color(l)[0] for l in labels]

    ax = f3_ax0
    width = 0.8

    ax.bar(bar_x - width / 2, bar_y, width, color=color_ranking)
    ax.set_xticks(bar_x - .5)
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], size=20, color='black')
    ax.set_title(
        "{} \nRF Feature Rankings".format(title),
        color='black', size=20)
    ax.set_xticklabels(hr_labels, rotation=60, color='black', ha='right',
                       fontsize=15)
    plt.subplots_adjust()

    if savefig:
        plt.savefig(savefig, format='pdf', dpi=300, transparent=True)

    plt.show()

def poly_fit_feature_rank_shade(forest, sorted_keys,
                                x_spectra_samples, title='', x_domain=None,
                          savefig='',
                          ):
    """
    Produce a feature ranking diagram for the polynomial fitted
    random forests.

    :param forest:
    :param sorted_keys:
    :param title:
    :return:
    """


    importances = list(forest.feature_importances_)

    # Match feature importances with names of features
    feature_importances = []
    for i, key in enumerate(sorted_keys):
        feature_importances.append((key, importances[i]))
    feature_importances = sorted(feature_importances, key=lambda x: x[1],
                                 reverse=True)
    mean_imp = np.mean(importances)
    max_imp = np.max(importances)

    top_importances = [(x[0], x[1] / max_imp) for x in feature_importances
                       if
                       x[1] > mean_imp][:15]
    if len(top_importances) < 14:
        print("Warning- top importances not that good")
    labels = [ti[0] for ti in top_importances]
    bar_x = np.arange(len(top_importances))
    bar_y = [ti[1] for ti in top_importances]

    color_keys = [(1, 0), (1, 1)]
    for i in range(4): color_keys.append((2, i))
    for i in range(8): color_keys.append((4, i))
    for i in range(10): color_keys.append((5, i))
    for i in range(20): color_keys.append((10, i))
    color_codex = {color_keys[i]: 'silver' if i % 2 == 0 else 'silver' for
                   i in
                   range(len(color_keys))}
    color_codex['peak'] = 'black'

    for i, l in enumerate(labels):
        if l == 'peak':
            color_codex['peak'] = color_ranking[i]
            continue
        elif l.lower() == 'random':
            continue

        cc_index = label_to_color_codex(l)
        if color_codex[cc_index] == 'gray' \
                or color_codex[cc_index] == 'silver':
            color_codex[label_to_color_codex(l)] = color_ranking[i]

    fig3 = plt.figure(figsize=(12, 6), constrained_layout=True, dpi=300)
    gs = fig3.add_gridspec(3, 4)

    f3_ax0 = fig3.add_subplot(gs[:2, :2])

    f3_ax1 = fig3.add_subplot(gs[0, 2])
    f3_ax2 = fig3.add_subplot(gs[0, 3])
    f3_ax3 = fig3.add_subplot(gs[1, 2])
    f3_ax4 = fig3.add_subplot(gs[1, 3])
    f3_ax5 = fig3.add_subplot(gs[2, 2:])
    all_axes = [f3_ax0, f3_ax1, f3_ax2, f3_ax3, f3_ax4, f3_ax5]

    for ax in all_axes:
        ax.spines['left'].set_linewidth(3)
        ax.spines['top'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)

    ideal_axes = [f3_ax1, f3_ax2, f3_ax3, f3_ax4, f3_ax5]

    split_vals = [[0, 50, 100],
                  [0, 25, 50, 75, 100],
                  [0, 20, 40, 60, 80, 100],
                  [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                  [i * 5 for i in range(20)]]


    split_domains = {0: [], 1: [], 2: [], 3: [], 4: []}

    for i, split_list in enumerate(split_vals):

        for j, x in enumerate(split_list):
            if j == len(split_list) - 1:
                break
            split_domains[i].append(x_domain[x:split_list[j + 1]])

    split_i = -1
    for ax, split_num in zip(ideal_axes, fine_splits):
        split_i +=1

        x_domains = split_domains[split_i]
        lw = 5

        to_plot = np.random.choice(x_spectra_samples, size=50)


        for j, cur_x_domain in x_domains:

            rect = patches.Rectangle((cur_x_domain[0],-.1),
                                     width=cur_x_domain[-1]-cur_x_domain[0],
                                     height=1.6)


        for spec in to_plot:
            ax.plot(x_spectra_samples[spec],lw=1,alpha=.5,color='black')



        ax.set_xlim(x_domain[0], x_domain[-1])
        ax.set_ylim(-.1, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axvline(fake_X[pi], color=color_codex['peak'], ls='--', lw=5)
        ax.set_title(split_to_title[split_num], color='black', size=13)

    if x_domain is not None:
        bottom_ticks = [-5, -5 + 2, -5 + 4, -5 + 6, -5 + 8, -5 + 10]
        f3_ax5.set_xticks(ticks=bottom_ticks)
        f3_ax5.set_xticklabels(_domain_to_label(x_domain))
        f3_ax5.set_xlabel('Energy (eV)')

    hr_labels = [label_to_hr_and_color(l)[0] for l in labels]

    ax = f3_ax0
    width = 0.8

    ax.bar(bar_x - width / 2, bar_y, width, color=color_ranking)
    ax.set_xticks(bar_x - .5)
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], size=20, color='black')
    ax.set_title(
        "{} \nRF Feature Rankings".format(title),
        color='black', size=20)
    ax.set_xticklabels(hr_labels, rotation=60, color='black', ha='right',
                       fontsize=15)
    plt.subplots_adjust()

    if savefig:
        plt.savefig(savefig, format='pdf', dpi=300, transparent=True)

    plt.show()


def label_to_hr_new(label,alt_labels):
    if label=='peak':
        if alt_labels is 1:
            return 'Peak Loc.'
        elif alt_labels is 2 or alt_labels is 3:
            return "Peak E."
        else:
            return 'Peak Location'
    fingerprint = label.split(',')
    fingerprint = [x.split(':') for x in fingerprint]
    fingerprint = {x[0]: x[1] for x in fingerprint}
    for key in ['deg', 'fraction_size', 'chunk', 'coef']:
        fingerprint[key] = int(fingerprint[key])

    if fingerprint['coef'] == 0:

        if alt_labels is 1:
            return 'Const. ($a_0$)'
        elif alt_labels is 2:
            return "Const."
        elif alt_labels is 3:
            return "$a_0$"
        return 'Constant ($a_0$)'

    if fingerprint['coef'] == 1:

        if alt_labels is 2:
            return 'Linear'
        elif alt_labels is 3:
            return '$a_1$'
        return 'Linear ($a_1$)'

    if fingerprint['coef'] == 2:
        if alt_labels is 1:
            return 'Quad. ($a_2$)'
        elif alt_labels is 2:
            return 'Quad.'
        elif alt_labels is 3:
            return "$a_2$"

        return 'Quadratic ($a_2$)'

    if fingerprint['coef'] == 3:
        if alt_labels is 2:
            return 'Cubic'
        elif alt_labels is 3:
            return '$a_3$'
        return 'Cubic ($a_3$)'


def fingerprint_to_split(label, pre_post=False):
    # Label expects form like: 'loc:all,deg:3,fraction_size:20,chunk:2,coef:0'
    fingerprint = label.split(',')
    fingerprint = [x.split(':') for x in fingerprint]
    fingerprint = {x[0]: x[1] for x in fingerprint}
    for key in ['deg', 'fraction_size', 'chunk', 'coef']:
        fingerprint[key] = int(fingerprint[key])

    # print(fingerprint)

    mapper1 = {1: 0,
               2: 1,
               4: 2,
               5: 3,
               10: 4,
               20: 5}
    return mapper1[fingerprint['fraction_size']], fingerprint['chunk']


def poly_fit_feature_rank_rainbow(forest, sorted_keys, x_domain,peak_locations=None,
                                  title='',
                                  savefig='',alt_labels=False,
                                  target_axes = None):
    y_ordering = np.linspace(1, 0, 15)

    """
    Produce a feature ranking diagram for the polynomial fitted
    random forests.

    :param forest:
    :param sorted_keys:
    :param title:
    :return:
    """
    split_vals = [[0, 100],
                  [0, 50, 100],
                  [0, 25, 50, 75, 100],
                  [0, 20, 40, 60, 80, 100],
                  [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                  [i * 5 for i in range(20 + 1)]]
    split_domains = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

    for i, split_list in enumerate(split_vals):

        for j, x in enumerate(split_list):
            if j == len(split_list) - 1:
                break
            split_domains[i].append(x_domain[x:split_list[j + 1]])
    importances = list(forest.feature_importances_)

    # Match feature importances with names of features
    feature_importances = []
    for i, key in enumerate(sorted_keys):
        feature_importances.append((key, importances[i]))
    feature_importances = sorted(feature_importances, key=lambda x: x[1],
                                 reverse=True)
    mean_imp = np.mean(importances)
    max_imp = np.max(importances)

    top_importances = [(x[0], x[1] / max_imp) for x in
                       feature_importances][:15]
    # if
    # x[1] > mean_imp][:15]
    if len(top_importances) < 14:
        print("Warning- top importances not that good")
    #pprint(top_importances)
    labels = [ti[0] for ti in top_importances]
    bar_x = np.arange(len(top_importances))
    bar_y = [ti[1] for ti in top_importances]

    if target_axes is None:
        fig = plt.figure(figsize=(2, 3), dpi=300, constrained_layout=True)
        gs = fig.add_gridspec(4, 1)
        ax0 = fig.add_subplot(gs[1:, 0])
        ax1 = fig.add_subplot(gs[0, 0])

    else:
        ax0 = target_axes[0]
        ax1 = target_axes[1]

    hr_labels = [label_to_hr_new(l,alt_labels=alt_labels) for l in labels]

    width = 0.8

    ax0.bar(bar_x - width / 2, bar_y, width, color=color_ranking)
    ax0.set_xticks(bar_x - .5)
    # ax0.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], size=20, color='black')
    ax0.set_yticklabels([], size=20, color='black')

    if target_axes is None:
        ax0.set_title(
            "{}\nRF Feature Rankings by Polynomial Coefficient".format(title),
            color='black', size=5)
    ax0.set_xticklabels(hr_labels, rotation=60, color='black', ha='right',
                        fontsize=5)
    # plt.subplots_adjust()

    the_xticks = np.linspace(x_domain[0],x_domain[-1],1000)
    the_xticks = sorted(list(set(list(np.around(the_xticks,-1)))))

    ax1.set_xticks(the_xticks)
    ax1.set_xticklabels(np.array(the_xticks).astype(int),
                        fontsize=5)
    ax1.set_xlim(x_domain[0], x_domain[-1])
    ax1.set_xlabel("Energy (eV)", fontsize=5)
    ax1.set_yticklabels(labels=[])
    ax1.set_ylim(-0.1, 1.1)
    if target_axes is None:
        ax1.set_title("Corresponding Regions", fontsize=8)
    else:
        ax1.set_title(title, color='black',size=8)
    # ax1.plot()
    for i in range(15):
        if top_importances[i][0]=='peak':
            ax1.plot([x_domain[peak_locations[0]],x_domain[peak_locations[1]]],
                     [y_ordering[i],y_ordering[i]],color=color_ranking[i])
            continue
        tgt_idxs = fingerprint_to_split(top_importances[i][0])
        #print(tgt_idxs)
        target_domain = split_domains[tgt_idxs[0]][tgt_idxs[1]]
        ax1.plot([target_domain[0], target_domain[-1]],
                 [y_ordering[i], y_ordering[i]], color=color_ranking[i])

    if savefig and target_axes is None:
        plt.savefig(savefig, format='pdf', dpi=300, transparent=True)

    if target_axes is None:
        plt.show()
