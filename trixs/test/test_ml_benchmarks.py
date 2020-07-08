# Copyright 2019-2020 Toyota Research Institute. All rights reserved.

import numpy as np
from trixs.machine_learning.benchmarks import precision_recall, precision_recall_matrix, confusion_dict


def test_precision_recall():
    """
    Compute precision and recall and F1 for a simple case
    Then ensure the precision recall matrix function does it's job
    :return:
    """

    truth = [0,0,1,1]
    guesses=[0,0,0,1]

    pr0  = precision_recall(guesses,truth,0)
    pr1 = precision_recall(guesses,truth,1)
    # Precision = true positives / FP + TP
    assert pr0[0] == 2 / (1+2)
    assert pr1[0] == 1/(0+1)

    # Recall = True Positives / (FN + TP)
    assert pr0[1] == 2 / (0 + 2)
    assert pr1[1] == 1/(1+1)

    # F1 = 2 * Pr * Rec / (Prec+Rec)

    assert pr0[2] == 2*pr0[0]*pr0[1] / (pr0[0]+pr0[1])
    assert pr1[2] == 2*pr1[0]*pr1[1] / (pr1[0]+pr1[1])

    assert (precision_recall_matrix(guesses,truth,[0,1]) == np.array([pr0,pr1])).all()


def test_confusion_dict():
    truth = [0,0,1,1]
    guesses = [0,0,0,1]

    conf_dict = confusion_dict(guesses,truth,[0,1])

    assert len(conf_dict) == 2

    assert conf_dict[0][0] == 2
    assert conf_dict[0][1] == 0
    assert conf_dict[1][0] == 1
    assert conf_dict[1][1] == 1



