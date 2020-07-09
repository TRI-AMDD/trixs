# Copyright 2019-2020 Toyota Research Institute. All rights reserved.

import numpy as np

from trixs.machine_learning.benchmarks import precision_recall, \
                            precision_recall_matrix, confusion_dict

def test_precision_recall():

    fits = [0, 1, 0, 0]
    labels = [0, 1, 1, 1]

    precision_0 = 1 / 3
    recall_0 = 1
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)

    precision_1 = 1
    recall_1 = 1 / 3
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)

    result_0 = precision_recall(fits, labels, 0)
    result_1 = precision_recall(fits, labels, 1)

    assert result_0[0] == precision_0
    assert result_0[1] == recall_0
    assert result_0[2] == f1_0


    assert result_1[0] == precision_1
    assert result_1[1] == recall_1
    assert result_1[2] == f1_1

    fits = [0,1,2]
    labels = [1,1,1]

    assert np.array_equal(precision_recall(fits, labels, 0),[0,0,0])
    assert np.array_equal(precision_recall(fits, labels, 1),[1,1/3,(2 *1/3 )
                                                             /(4/3)])
    assert np.array_equal(precision_recall(fits, labels, 2),[0,0,0])


def test_precision_recall_matrix():


    fits = [0,1,2]
    labels = [1,1,1]

    answer = [[0,0,0], [1,1/3,(2 *1/3 )/(4/3)], [0,0,0]]

    assert np.array_equal(precision_recall_matrix(fits,labels,[0,1,2]),answer)


def test_confusion_dict():

    fits = [0,1,2,3]
    labels = [1,1,1,2]

    conf_dict = confusion_dict(fits,labels,classes=[0,1,2,3])

    assert conf_dict[1] == [1,1,1,0]

    assert conf_dict[2] == [0,0,0,1]







