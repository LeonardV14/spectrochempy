# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT 
# See full LICENSE agreement in the root directory
# ======================================================================================================================


""" Tests for the ndplugin module

"""

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core import general_preferences as prefs, log
import os

from spectrochempy.utils.testing import assert_array_equal


# Basic
# ----------------------------------------------------------------------------------------------------------------------
def test_basic():
    ir = NDDataset([1.1, 2.2, 3.3], coords=[[1, 2, 3]])
    ir.save('essai')
    dl = NDDataset.load('essai')
    assert_array_equal(dl.data, ir.data)

    ir = NDDataset([[1.1, 2.2, 3.3]], coords=[[0], [1, 2, 3]])
    ir.save('essai')
    dl = NDDataset.load('essai')
    assert_array_equal(dl.data, ir.data)

    ir = NDDataset([[1.1, 2.2, 3.3], [1.1, 2.2, 3.3]], coords=[[1, 2], [1, 2, 3]])
    ir.save('essai')
    dl = NDDataset.load('essai')
    assert_array_equal(dl.data, ir.data)


def test_save1D_load(IR_dataset_1D):
    dataset = IR_dataset_1D.copy()
    log.debug(dataset)
    dataset.save('essai')
    ir = NDDataset.load("essai")
    log.debug(ir)
    os.remove(os.path.join(prefs.datadir, 'essai.scp'))


def test_save2D_load(IR_dataset_2D):
    dataset = IR_dataset_2D.copy()
    log.debug(dataset)
    dataset.save('essai')
    ir = dataset.load("essai")
    log.debug(ir)
    os.remove(os.path.join(prefs.datadir, 'essai.scp'))


def test_save_and_load_mydataset(IR_dataset_2D):
    ds = IR_dataset_2D.copy()
    ds.save('mydataset')
    dl = NDDataset.load('mydataset')
    assert_array_equal(dl.data, ds.data)
    assert_array_equal(dl.x.data, ds.x.data)
    assert (dl == ds)
    assert (dl.meta == ds.meta)
    assert (dl.plotmeta == ds.plotmeta)
