# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
In this module, we define basic functions adapted from numpy but able to handle
our NDDataset objects

"""
__all__ = ['diag', 'dot', 'empty', 'empty_like', 'zeros', 'zeros_like', 'ones',
           'ones_like', 'full', 'full_like']


# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

import numpy as np
from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.utils import nomask

def empty(shape, dtype=None, **kwargs):
    """
    Return a new |NDDataset| of given shape and type,  without initializing
    entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array
    dtype : data-type, optional
        Desired output data-type.

    Returns
    -------
    out : |NDDataset|
        Array of uninitialized (arbitrary) data of the given shape, dtype, and
        order.  Object arrays will be initialized to None.

    See Also
    --------
    empty_like, zeros, ones

    Notes
    -----
    `empty`, unlike `zeros`, does not set the array values to zero,
    and may therefore be marginally faster.  On the other hand, it requires
    the user to manually set all the values in the array, and should be
    used with caution.

    Examples
    --------
    >>> from spectrochempy import core # doctest: +ELLIPSIS
    SpectroChemPy's API...

    >>> core.empty([2, 2]) # doctest: +ELLIPSIS
    NDDataset: [[...]] unitless

    >>> core.empty([2, 2], dtype=int, units='s') # doctest: +ELLIPSIS
    NDDataset: [[...]] s

    """
    return NDDataset(np.empty(shape, dtype=dtype), **kwargs)

def empty_like(a, dtype=None):
    """
    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of the
        returned array.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data with the same
        shape and type as `a`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.

    Notes
    -----
    This function does *not* initialize the returned array; to do that use
    for instance `zeros_like`, `ones_like` or `full_like` instead.  It may be
    marginally faster than the functions that do set the array values.

    """
    new = a.copy()
    return new

def zeros(shape, dtype=None, **kwargs):
    """
    Return a new |NDDataset| of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    **kwargs : keyword args to pass to the |NDDataset| constructor

    Returns
    -------
    out : |NDDataset|
        Array of zeros with the given shape, dtype.

    See Also
    --------
    ones, zeros_like

    Examples
    --------
    >>> from spectrochempy import core
    >>> core.zeros(5)
    NDDataset: [   0.000,    0.000,    0.000,    0.000,    0.000] unitless

    >>> core.zeros((5,), dtype=np.int)
    NDDataset: [       0,        0,        0,        0,        0] unitless

    >>> s = (2,2)
    >>> core.zeros(s, units='m')
    NDDataset: [[   0.000,    0.000],
                [   0.000,    0.000]] m

    """
    return NDDataset(np.zeros(shape, dtype=dtype), **kwargs)


def ones(shape, dtype=None, **kwargs):
    """
    Return a new |NDDataset| of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    **kwargs : keyword args to pass to the |NDDataset| constructor

    Returns
    -------
    out : |NDDataset|
        Array of ones with the given shape, dtype.

    See Also
    --------
    zeros, ones_like

    Examples
    --------
    >>> from spectrochempy import core
    >>> core.ones(5, units='km')
    NDDataset: [   1.000,    1.000,    1.000,    1.000,    1.000] km

    >>> core.ones((5,), dtype=np.int, mask=[True, False, False, False, True])
    NDDataset: [  --,        1,        1,        1,   --] unitless

    >>> core.ones((2, 2))
    NDDataset: [[   1.000,    1.000],
                [   1.000,    1.000]] unitless

    """
    return NDDataset(np.ones(shape, dtype=dtype), **kwargs)


def zeros_like(a, dtype=None,):
    """
    Return a |NDDataset| of zeros with the same shape and type as a given
    array.

    Parameters
    ----------
    a : |NDDataset|
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    out : |NDDataset|
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> from spectrochempy import core
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x = NDDataset(x, units='s')
    >>> x
    NDDataset: [[       0,        1,        2],
                [       3,        4,        5]] s
    >>> core.zeros_like(x)
    NDDataset: [[       0,        0,        0],
                [       0,        0,        0]] s


    """
    new = a.copy()
    new.data = np.zeros_like(a)
    return new


def ones_like(a, dtype=None):
    """
    Return |NDDataset| of ones with the same shape and type as a given array.

    It preserves original mask, units, coords and uncertainty

    Parameters
    ----------
    a : |NDDataset|
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    out : |NDDataset|
        Array of ones with the same shape and type as `a`.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> from spectrochempy import core
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x = NDDataset(x, units='s')
    >>> x
    NDDataset: [[       0,        1,        2],
                [       3,        4,        5]] s
    >>> core.ones_like(x)
    NDDataset: [[       1,        1,        1],
                [       1,        1,        1]] s

    """
    new = a.copy()
    new.data = np.ones_like(a)
    return new

def full(shape, fill_value, dtype=None, **kwargs):
    """
    Return a new |NDDataset| of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `np.int8`.  Default
        is `float`, but will change to `np.array(fill_value).dtype` in a
        future release.
    **kwargs : keyword args to pass to the |NDDataset| constructor

    Returns
    -------
    out : |NDDataset|
        Array of `fill_value` with the given shape, dtype, and order.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    full_like : Fill an array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> from spectrochempy import core
    >>> core.full((2, 2), np.inf)
    NDDataset: [[     inf,      inf],
                [     inf,      inf]] unitless
    >>> core.full((2, 2), 10, dtype=np.int)
    NDDataset: [[      10,       10],
                [      10,       10]] unitless

    """
    return NDDataset(np.full(shape, fill_value=fill_value, dtype=dtype),
                     **kwargs)

def full_like(a, fill_value, dtype=None):
    """
    Return a |NDDataset| with the same shape and type as a given array.

    Parameters
    ----------
    a : |NDDataset| or array-like
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    array-like
        Array of `fill_value` with the same shape and type as `a`.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.
    full : Fill a new array.

    Examples
    --------
    >>> from spectrochempy import core

    >>> x = np.arange(6, dtype=int)
    >>> core.full_like(x, 1)
    array([       1,        1,        1,        1,        1,        1])

    >>> x = NDDataset(x, units='m')
    >>> core.full_like(x, 0.1)
    NDDataset: [       0,        0,        0,        0,        0,        0] m
    >>> core.full_like(x, 0.1, dtype=np.double)
    NDDataset: [   0.100,    0.100,    0.100,    0.100,    0.100,    0.100] m
    >>> core.full_like(x, np.nan, dtype=np.double)
    NDDataset: [     nan,      nan,      nan,      nan,      nan,      nan] m

    """
    new = a.copy()
    new.data = np.full_like(a, fill_value=fill_value, dtype=dtype)
    return new


# ............................................................................
def dot(a, b, strict=True, out=None):
    """
    Return the dot product of two NDDatasets.

    This function is the equivalent of `numpy.dot` that takes NDDataset for
    input.

    .. note::
      Works only with 2-D arrays at the moment.


    Parameters
    ----------
    a, b : masked_array_like
        Inputs arrays.
    strict : bool, optional
        Whether masked data are propagated (True) or set to 0 (False) for
        the computation. Default is False.  Propagating the mask means that
        if a masked value appears in a row or column, the whole row or
        column is considered masked.
    out : masked_array, optional
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right type, must be
        C-contiguous, and its dtype must be the dtype that would be returned
        for `dot(a,b)`. This is a performance feature. Therefore, if these
        conditions are not met, an exception is raised, instead of attempting
        to be flexible.

    See Also
    --------
    numpy.dot : Equivalent function for ndarrays.
    numpy.ma.dot : Equivalent function for masked ndarrays

    """
    if not a.implements('NDDataset'):
        raise TypeError('A dataset of type NDDataset is  '
                        'expected as a source of data, but an object'
                        ' of type {} has been provided'.format(
            type(a).__name__))

    if not b.implements('NDDataset'):
        raise TypeError('A dataset of type NDDataset is  '
                        'expected as a source of data, but an object'
                        ' of type {} has been provided'.format(
            type(b).__name__))

    #TODO: may be we can be less strict, and allow dot products with
    # different kind of objects, as far they are numpy-like arrays

    data = np.ma.dot(a.masked_data, b.masked_data)
    mask = data.mask
    data = data.data
    uncertainty = None
    if a.is_uncertain or b.is_uncertain:
            raise NotImplementedError('uncertainty not yet implemented')

    coordset = None
    if a.coordset is not None:
        coordset = [a.coordset[0]]
    if b.coordset is not None:
        if coordset is None:
            coordset = [None]
        coordset.append(b.coordset[1])
    elif coordset is not None:
        coordset.append(None)

    history = 'dot product between %s and %s'%(a.name, b.name)

    # make the output
    # ---------------
    new = a.copy()
    new._data = data
    new._mask = mask
    new._uncertainty = uncertainty
    new._coordset =  type(new.coordset)(coordset)
    new.history = history

    return new


# ............................................................................
def diag(source, k=0):
    """
    Extract a diagonal or construct a diagonal array.

    See the more detailed documentation for ``numpy.diagonal`` if you use this
    function to extract a diagonal and wish to write to the resulting array;
    whether it returns a copy or a view depends on what version of numpy you
    are using.

    Parameters
    ----------
    v : array_like
        If `v` is a 2-D array, return a copy of its `k`-th diagonal.
        If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
        diagonal.
    k : int, optional
        Diagonal in question. The default is 0. Use `k>0` for diagonals
        above the main diagonal, and `k<0` for diagonals below the main
        diagonal.

    Returns
    -------
    out : ndarray
        The extracted diagonal or constructed diagonal array.

    copied from numpy (licence
    """

    # check if we have the correct input
    # ----------------------------------

    if not source.implements('NDDataset'):
        raise TypeError('A dataset of type NDDataset is  '
                        'expected as a source of data, but an object'
                        ' of type {} has been provided'.format(
            type(source).__name__))

    s = source.data.shape

    if len(s) == 1:
        # construct a diagonal array
        # --------------------------
        data = np.diag(source.data)
        mask = nomask
        if source.is_masked:
            size = source.size
            m = np.repeat(source.mask, size).reshape(size, size)
            mask = m | m.T
        uncertainty = None
        if source.is_uncertain:
            uncertainty = np.diag(source.uncertainty)
        coordset = None
        if source.coordset is not None:
            coordset = [source.coordset[0]]*2
        history = 'diagonal array build from the 1D dataset'

    elif len(s) == 2:
        # extract a diagonal
        # ------------------
        data = np.diagonal(source.data, k).copy()
        mask = None
        if source.is_masked:
            mask = np.diagonal(source.mask, k).copy()
        uncertainty = None
        if source.is_uncertain:
            uncertainty = np.diagonal(source.uncertainty, k).copy()
        coordset = None
        if source.coordset is not None:
            coordset = [source.coordset[0]]  # TODO: this is likely not
                                             #       correct for k != 0
        history = 'diagonal of rank %d extracted from original dataset'%k

    else:
        raise ValueError("Input must be 1- or 2-d.")

    # make the output
    # ---------------
    new = source.copy()
    new._data = data
    new._mask = mask
    new._uncertainty = uncertainty
    new.coordset = coordset
    new.history = history

    return new



# =============================================================================
if __name__ == '__main__':
    pass