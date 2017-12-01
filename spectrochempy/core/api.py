# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
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

# here we load all sub packages routines

import sys
from traitlets import import_item

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import list_packages

pkgs = sys.modules['spectrochempy.core']
api = sys.modules['spectrochempy.core.api']

pkgs = list_packages(pkgs)

__all__ = []

# dataset
# --------
from spectrochempy.core.dataset.api import *
from spectrochempy.core.dataset import api

__all__ += api.__all__

# plotters
# --------
from spectrochempy.core.plotters.api import *
from spectrochempy.core.plotters import api

__all__ += api.__all__

# processors
# ------------
from spectrochempy.core.processors.api import *
from spectrochempy.core.processors import api

__all__ += api.__all__

# readers
# ------------
from spectrochempy.core.readers.api import *
from spectrochempy.core.readers import api

__all__ += api.__all__

# writers
# ------------
from spectrochempy.core.writers.api import *
from spectrochempy.core.writers import api

__all__ += api.__all__

# units
# ------------
from spectrochempy.core.units import *
from spectrochempy.core import units

__all__ += units.__all__
