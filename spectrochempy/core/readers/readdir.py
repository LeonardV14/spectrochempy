# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""This module to extend NDDataset with the import methods method.

"""
__all__ = ['read_dir']

__dataset_methods__ = __all__

# ----------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------

import os
import warnings
import tkinter as tk
from tkinter import filedialog

# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------------
# local imports
# ----------------------------------------------------------------------------

from spectrochempy.dataset.ndio import NDIO
from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.application import datadir, log
from spectrochempy.utils import readfilename, SpectroChemPyWarning
from .readomnic import read_omnic
from .readcsv import read_csv


# function for reading data in a directory
# --------------------------------------
def read_dir(dataset=None, directory=None, **kwargs):
    """Open readable files in a directory and store
    data/metadata in a dataset or a list of datasets according to the
    following rules:
    2d spectroscopic data (e.g. valid .spg files)
    from distinct files are stored in distinct NDdatasets. 1d spectroscopic data (e.g.
    .spa files) in a given directory are grouped into single NDDatasets, providing their unique dimensions are
     compatible. Only implemented for omnic files (spa, spg).

    Parameters
    ----------
    :param: dataset : `NDDataset`
        The dataset to store the data and metadata. If None, a NDDataset is created
    :param: directory: str . If not specified, opens a dialog box.
    TODO add :param: recursive: bool [optional default = True ]. read also subfolders
    :param: sortbydate: bool [optional default = True]. sort spectra by acquisition date
    :returns: a 'NDDdataset' or a list of 'NDdatasets'.


    Examples
    --------
    >>> from spectrochempy import NDDataset # doctest: +ELLIPSIS,
    +NORMALIZE_WHITESPACE
    SpectroChemPy's API ...
    >>> A = NDDataset.read_dir('irdata')
    >>> print(A) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    <BLANKLINE>
    --------------------------------------------------------------------------------
      name/id: nh4y-activation.spg ...
    >>> B = NDDataset.read_dir()

    """
    log.debug("starting read_dir()")

    # check if the first parameter is a dataset
    # because we allow not to pass it
    if not isinstance(dataset, NDDataset):
        # probably did not specify a dataset
        # so the first parameter must be the directory
        if isinstance(dataset, str) and dataset != '':
            directory = dataset

        dataset = NDDataset()  # create a NDDataset

    # check directory
    if not directory:
        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.geometry('0x0+0+0')
        root.deiconify()
        root.lift()
        root.focus_force()
        directory = filedialog.askdirectory(parent=root)

        root.quit()

    if not isinstance(directory, str):
        raise TypeError('Error: directory should be of str type')

    datasets = []

    recursive = kwargs.get('recursive', True)
    if recursive:
        for i,root in enumerate(os.walk(directory)):
            if i==0:
                log.debug("reading main directory")
            else:
                log.debug("reading subdirectory")
            datasets += _read_single_dir(root[0])
    else:
        log.debug("reading directory")
        datasets += _read_single_dir(directory)

    if len(datasets)==1:
        log.debug("finished read_dir()")
        return datasets[0] # a single dataset is returned
    log.debug("finished read_dir()")
    return datasets  # several datasets returned

def _read_single_dir(directory):
    # lists all filenames of readable files in directory:
    filenames = [os.path.join(directory, f) for f in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, f))]

    datasets = []

    if not filenames:
        return datasets

    files = readfilename(filenames,
                         directory = directory)

    for extension in files.keys():
        if extension == '.spg':
            for filename in files[extension]:
                datasets.append(read_omnic(filename))

        elif extension == '.spa':
            dataset = NDDataset()
            datasets.append(read_omnic(dataset, files[extension],
                                     sortbydate=True))

        #TODO: uncomment below and test .csv
        # elif extension == '.csv':
        #    datasets.append(read_csv(dataset, files[extension])

        # else the files are not readable
        else:
            pass

        #TODO: extend to other implemented readers
    return datasets

if __name__ == '__main__':

    pass
