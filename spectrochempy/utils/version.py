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

import os
import subprocess
import setuptools_scm
from pkg_resources import get_distribution, DistributionNotFound

# .............................................................................
def get_version():

    root = os.path.join(os.path.dirname(__file__), '../..')
    try:
        # let's first try to get version from git
        version = setuptools_scm.get_version(
                version_scheme='post-release',
                root=root,
                relative_to=__file__).split('+')[0]
    except:
        try:
            # let's try with the distribution version
            version = get_distribution('spectrochempy').version

        except DistributionNotFound:

            from spectrochempy.version import version

    path = os.path.join(root, 'spectrochempy', 'version.py')
    with open(path, "w") as f:
        f.write("version = '%s' " % version)

    return version

def get_release():

    version = get_version()
    release = version.split('.post')[0]
    return release

# .............................................................................
def get_release_date():
    try:
        return subprocess.getoutput(
            "git log -1 --tags --date='short' --format='%ad'")
    except:
        pass

# .............................................................................
def get_version_date():
    try:
        return subprocess.getoutput(
            "git log -1 --date='short' --format='%ad'")
    except:
        pass



# =============================================================================
# __main__
# =============================================================================
if __name__ == '__main__':

    print("release :", get_release())
    print("release data: ",get_release_date())

    print("version :", get_version())
    print("version date: ", get_version_date())


