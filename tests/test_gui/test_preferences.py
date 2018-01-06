# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

from spectrochempy.gui.preferences import (Preferences,
                                           GeneralOptionsWidget,
                                           ProjectOptionsWidget,
                                           PlotOptionsWidget)

from spectrochempy.extern.pyqtgraph import mkQApp

app = mkQApp()

class testPreferences():

    def __init__(self):

        self.preference_pages = [GeneralOptionsWidget,
                                      ProjectOptionsWidget,
                                      PlotOptionsWidget]

        self.preferences = dlg = Preferences()

        for Page in self.preference_pages:
            page = Page(dlg)
            page.initialize()
            dlg.add_page(page)

        dlg.exec_()

tp = testPreferences()


# =============================================================================
if __name__ == '__main__':
    pass
