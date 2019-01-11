# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

""" Tests for the PCA module

"""
from spectrochempy import PCA, masked

# test pca
#---------

def test_pca(IR_dataset_2D):

    dataset = IR_dataset_2D.copy()

    # with masks
    dataset[:, 1240.0:920.0] = masked  # do not forget to use float in slicing

    pca = PCA(dataset)
    print(pca)
    pca.printev(n_pc=5)

    assert str(pca)[:3] == '\nPC'

    pca.screeplot(npc=0.95)

    pca.screeplot(npc='auto')

    pca.scoreplot((1,2))

    pca.scoreplot(1,2, 3)
