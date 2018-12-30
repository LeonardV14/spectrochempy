# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# =============================================================================




""" Tests for the  module

"""
import sys
import functools
import pytest
from spectrochempy.utils.testing import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises)

from spectrochempy import *

from spectrochempy.utils import SpectroChemPyWarning




def test_nmr_2D_em_(NMR_dataset_2D):
    dataset = NMR_dataset_2D.copy()
    dataset.plot()
    assert dataset.shape == (96, 948)
    dataset.em(lb=100. * ur.Hz)
    assert dataset.shape == (96, 948)
    dataset.em(lb=50. * ur.Hz, axis=0)
    assert dataset.shape == (96, 948)
    dataset.plot(cmap='copper', data_only=True)
    show()
    pass
