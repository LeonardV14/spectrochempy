# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================


""" Tests for the  module

"""
import sys
import functools
import pytest
from tests.utils import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises, show_do_not_block)


from spectrochempy.api import *
from spectrochempy.api import figure, show
from spectrochempy.utils import SpectroChemPyWarning


# nmr_processing
#-----------------------------
@show_do_not_block
def test_nmr_1D_show(NMR_source_1D):
    source = NMR_source_1D.copy()
    figure()
    ax1 = source.plot()
    assert ax1 is not None
    assert source.is_complex[-1]
    show()
    pass

@show_do_not_block
def test_nmr_1D_show_hold(NMR_source_1D):
    source = NMR_source_1D.copy()
    figure()
    # test if we can plot on the same figure
    source.plot(xlim=(0.,25000.))
    # we want to superpose a second spectrum
    source.plot(imag=True, data_only=True)
    show()

@show_do_not_block
def test_nmr_1D_show_dualdisplay(NMR_source_1D):
    source = NMR_source_1D.copy()
    # test if we can plot on the same figure
    source.plot(xlim=(0.,25000.))
    source.em(lb=100. * ur.Hz)
    # we want to superpose a second spectrum
    source.plot()
    show()

    figure()
    source.plot()
    show()

@show_do_not_block
def test_nmr_1D_show_dualdisplay_apodfun(NMR_source_1D):
    source = NMR_source_1D.copy()
    figure()
    # test if we can plot on the same figure
    source.plot(xlim=(0.,25000.))
    # we want to superpose a second spectrum wich is the apodization function
    LB = 80 * ur.Hz
    source.em(lb=LB)
    source.plot(data_only=True)
    # display the apodization function
    apodfun = source.em(lb=LB, apply=False)
    apodfun.plot(data_only=True)
    show()

@show_do_not_block
def test_nmr_1D_show_complex(NMR_source_1D):
    # display the real and complex at the same time
    source = NMR_source_1D.copy()
    source.plot(show_complex=True, color='green',
                xlim=(0.,30000.), zlim=(-2.,2.))
    show()

def test_nmr_em_nothing_calculated(NMR_source_1D_1H):
    # em without parameters
    source = NMR_source_1D_1H.copy()

    arr = source.em(apply=False)
    # we should get an array of ones only , as apply = False mean
    # that we do not apply the apodization, but just make the
    # calculation of the apodization function
    assert_equal(arr, np.ones_like(source.data))

def test_nmr_em_calculated_notapplied(NMR_source_1D_1H):
    # em calculated but not applied
    source = NMR_source_1D_1H.copy()

    lb = 100
    arr = source.em(lb=lb, apply=False)
    assert isinstance(arr, NDDataset)

    # here we assume it is 100 Hz
    x = source.axes[-1]
    tc = (1./(lb * ur.Hz)).to(x.units)
    e = np.pi * np.abs(x) / tc
    arrcalc = np.exp(-e.data)

    assert_equal(arr.real().data, arrcalc)  # note that we have to compare
    # to the real part data because of the complex nature of the data

def test_nmr_em_calculated_applied(NMR_source_1D_1H):
    # em calculated and applied
    source = NMR_source_1D_1H.copy()

    lb = 100
    arr = source.em(lb=lb, apply=False)

    # here we assume it is 100 Hz
    x = source.axes[-1]
    tc = (1. / (lb * ur.Hz)).to(x.units)
    e = np.pi * np.abs(x) / tc
    arrcalc = np.exp(-e.data)

    # check with apply = True (by default)
    source2 = source.copy()
    source3 = source.em(lb=lb)

    # data should be equal
    assert_equal(source3.data, (arrcalc*source2).data)

    # but also the sources as whole entity
    assert(source3 == arrcalc*source2)

def test_nmr_em_calculated_Hz(NMR_source_1D_1H):
    source = NMR_source_1D_1H.copy()

    lb = 200 * ur.Hz
    x = source.axes[-1]
    tc = (1. / lb).to(x.units)
    e = np.pi * np.abs(x) / tc
    arrcalc = np.exp(-e.data)

    source2 = source.copy()
    source3 = source.em(lb=lb, inplace=False)

    # the sources should be equal
    assert(source3 == arrcalc*source2)

    # and the original untouched
    assert (source != source3)

def test_nmr_em_calculated_inplace(NMR_source_1D_1H):
    source = NMR_source_1D_1H.copy()

    lb = 200 * ur.Hz

    x = source.axes[-1]
    tc = (1. / lb).to(x.units)
    e = np.pi * np.abs(x) / tc
    arrcalc = np.exp(-e.data)

    source2 = source.copy()
    source.em(lb=lb)  # inplace transformation

    # the sources data array should be equal
    s = arrcalc * source2
    assert(np.all(source.data == s.data))

    # as well as the whole new sources
    assert (source == arrcalc * source2)


@show_do_not_block
def test_nmr_1D_em_(NMR_source_1D_1H):

    source = NMR_source_1D_1H.copy()

    source.plot(xlim=(0.,6000.))

    source.em(lb=100.*ur.Hz)

    source.plot(data_only=True)

    # successive call
    source.em(lb=200. * ur.Hz)

    source.plot(data_only=True)

    show()

@show_do_not_block
def test_nmr_1D_em_with_no_kw_lb_parameters(NMR_source_1D_1H):

    source = NMR_source_1D_1H.copy()

    source.plot()
    source.em(100.*ur.Hz, inplace=True)
    source.plot()
    show()

@show_do_not_block
def test_nmr_1D_em_inplace(NMR_source_1D_1H):
    source = NMR_source_1D_1H.copy()

    source.plot()
    source1 = source.em(lb=100. * ur.Hz)
    assert source1 is source # inplace transform by default
    try:
        assert_array_equal(source1.data, source.data)
    except AssertionError:
        pass
    show()

@show_do_not_block
def test_nmr_1D_gm(NMR_source_1D_1H):

    # first test gm
    source = NMR_source_1D_1H.copy()

    source.plot(xlim=(0.,6000.))

    source.gm(lb=100.*ur.Hz, gb=100.*ur.Hz)

    source.plot()
    show()

# def test_zf():
#     td = source1.meta.td[-1]
#     source1 = source1.zf(size=2*td)
#     #si = source_em.meta.si[-1]
#     source1.plot(hold=True)
#
#     # source1 = source1.fft()
#     source1.plot()
#     pass


#### TEST IN 2D #####
@show_do_not_block
def test_nmr_2D(NMR_source_2D):

    figure()
    source = NMR_source_2D
    source.plot()
    show()
    pass

@show_do_not_block
def test_nmr_2D_imag(NMR_source_2D):

    #plt.ion()
    figure()
    source = NMR_source_2D.copy()
    source.plot()
    source.plot(imag=True, cmap='jet', data_only=True)
                                # better not to replot a second colorbar
    show()
    pass

@show_do_not_block
def test_nmr_2D_hold(NMR_source_2D):

    source = NMR_source_2D
    figure()
    source.plot()
    source.imag().plot(cmap='jet', data_only=True)
    show()
    pass

@show_do_not_block
def test_nmr_2D_em_(NMR_source_2D):
    figure()
    source = NMR_source_2D.copy()
    source.plot()
    source.em(lb=20.*ur.Hz)
    source.em(lb=10. * ur.Hz, axis=0)
    source.plot(cmap='copper',data_only=True)
    show()
    pass