# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
This module define the `application` on which the API rely.


"""

# ============================================================================
# standard library import
# ============================================================================

import os
import glob
import sys
import logging
import warnings
import subprocess
import datetime
import warnings

# ============================================================================
# third party imports
# ============================================================================

from pkg_resources import get_distribution, DistributionNotFound
from setuptools_scm import get_version
from traitlets.config.configurable import Configurable
from traitlets.config.application import Application, \
    catch_config_error
from traitlets import (Instance, Bool, Unicode, List, Dict, default,
                          observe,
                       import_item, HasTraits)

import matplotlib as mpl
from matplotlib import pyplot as plt

from IPython import get_ipython
from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic,
                                line_cell_magic)
from IPython.core.magics.code import extract_symbols
from IPython.core.error import UsageError
from IPython.utils.text import get_text_list

# ============================================================================
# constants
# ============================================================================

__all__ = ['app']


# Log levels
# -----------------------------------------------------------------------------
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# ----------------------------------------------------------------------------
# Version
# ----------------------------------------------------------------------------

try:
    #: the release string of this package
    __release__ = get_distribution('spectrochempy').version
except DistributionNotFound:
    # package is not installed
    __release__ = '0.1.alpha'

try:
    #: the version string of this package
    __version__ = get_version(root='..', relative_to=__file__)
except:
    __version__ = __release__


# ............................................................................
def _get_copyright():
    current_year = datetime.date.today().year
    copyright = '2014-{}'.format(current_year)
    copyright += ' - A.Travert and C.Fernandez @ LCS'
    return copyright

#: the copyright string of this package
__copyright__ = _get_copyright()

# .............................................................................
def _get_release_date():
    try:
        return subprocess.getoutput(
            "git log -1 --tags --date='short' --format='%ad'")
    except:
        pass

#: the last release date of this package
__release_date__ = _get_release_date()

# ............................................................................
# other info

#: url for the documentation of this package
__url__ = "http://www-lcs.ensicaen.fr/spectrochempy"

#: first authors(s) of this package
__author__ = "C. Fernandez & A. Travert @LCS"

#: contributor(s) to this package
__contributor__ = ""

#: The license of this package
__license__ = "CeCILL-B license"




# ============================================================================
# Magic ipython function
# ============================================================================
@magics_class
class SpectroChemPyMagics(Magics):

    @line_cell_magic
    def addscript(self, pars='', cell=None):
        """This works both as **%addscript** and as **%%addscript**

        This magic command can either take a local filename, element in the
        namespace or history range (see %history),
        or the current cell content


        Usage
            %addscript  -p project  n1-n2 n3-n4 ... n5 .. n6 ...

             or

            %%addscript -p project
            ...code lines ...


        Options
            -p <string>         Name of the project where the script will be stored.
                                If not provided, a project with a standard name:
                                ``proj`` is searched.
            -o <string>         script name

            -s <symbols>        Specify function or classes to load from python
                                source.

            -a                  append to the current script instead of
                                overwriting it.

            -n                  search symbol in the current namespace


        Examples
        --------

        .. sourcecode:: ipython::

            In[1]: %addscript myscript.py

            In[2]: %addscript 7-27

            In[3]: %addscript -s MyClass,myfunction myscript.py

            In[4]: %addscript MyClass

            In[5]: %addscript mymodule.myfunction


        """
        opts, args = self.parse_options(pars, 'p:o:s:n:a')
        #print(opts)
        #print(args)
        #print(cell)

        append = 'a' in opts
        mode = 'a' if append else 'w'
        search_ns = 'n' in opts

        if not args and not cell and not search_ns:
            raise UsageError('Missing filename, input history range, '
                             'or element in the user namespace.\n '
                             'If no argument are given then the cell content '
                             'should '
                             'not be empty')
        name = 'script'
        if 'o' in opts:
            name = opts['o']

        proj = 'proj'
        if 'p' in opts:
            proj = opts['p']
        if not proj in self.shell.user_ns:
            raise ValueError('Cannot find any project with name `{}` in the '
                  'namespace.'.format(proj))
        # get the proj object
        projobj = self.shell.user_ns[proj]

        contents = ""
        if search_ns:
            contents += "\n" + self.shell.find_user_code(opts['n'],
                                                    search_ns=search_ns) + "\n"

        args = " ".join(args)
        if args.strip():
            contents += "\n" + self.shell.find_user_code(args,
                                                    search_ns=search_ns) + "\n"

        if 's' in opts:
            try:
                blocks, not_found = extract_symbols(contents, opts['s'])
            except SyntaxError:
                # non python code
                logging.error("Unable to parse the input as valid Python code")
                return

            if len(not_found) == 1:
                warnings.warn('The symbol `%s` was not found' % not_found[0])
            elif len(not_found) > 1:
                warnings.warn('The symbols %s were not found' % get_text_list(
                    not_found, wrap_item_with='`'))

            contents = '\n'.join(blocks)

        if cell:
            contents += "\n" + cell

        from spectrochempy.scripts.script import Script
        script = Script(name, content=contents)
        projobj[name]=script

        return "Script {} created.".format(name)

    @line_magic
    def runscript(self, pars=''):
        """

        """
        opts, args = self.parse_options(pars, '')

        if not args:
            raise UsageError('Missing script name')

        return args

# ==============================================================================
# SCPData class
# ==============================================================================

class SCPData(HasTraits):
    """
    This class is used to determine the path to the scp_data directory.

    """

    data = Unicode(help="Directory where to look for data")

    _data = Unicode()

    # ------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------

    def listing(self):
        """
        Create a str representing a listing of the data repertory.

        Returns
        -------
        listing : str

        """
        s = os.path.basename(self.data) + "\n"

        def _listdir(s, initial, ns):
            ns += 1
            for f in glob.glob(os.path.join(initial, '*')):
                fb = os.path.basename(f)
                if not fb.startswith('acqu') and \
                        not fb.startswith('pulse') and fb not in ['ser', 'fid']:
                    s += "   " * ns + "|__" + "%s\n" % fb
                if os.path.isdir(f):
                    s = _listdir(s, f, ns)
            return s

        return _listdir(s, self.data, -1)

    # ------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------

    def __str__(self):
        return self.listing()

    # ------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------

    @default('data')
    def _get_data_default(self):
        # return the spectra dir by default
        return self._data

    @default('_data')
    def _get__data_default(self):
        # the spectra path in package data
        return self._get_pkg_data_dir('testdata', 'scp_data')


    # ------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------

    def _repr_html_(self):
        # _repr_html is needed to output in notebooks
        return self.listing().replace('\n', '<br/>').replace(" ", "&nbsp;")

    def _get_pkg_data_dir(self, data_name, package=None) :

        data_name = os.path.normpath(data_name)

        datadir = os.path.dirname(import_item(package).__file__)
        datadir = os.path.join(datadir, data_name)

        if not os.path.isdir(datadir) :
            return os.path.dirname(datadir)

        return datadir


# ============================================================================
# Main application and configurators
# ============================================================================
class GeneralPreferences(Configurable) :
    """Preferences that apply to the |scp| application in general"""

    _scpdata = Instance(SCPData,
                    help="Set a data directory where to look for data"
                    )

    @default('_scpdata')
    def _get__scpdata_default(self):
        return SCPData()

    @property
    def list_scpdata(self):
        return self._scpdata


    # configuration parameters
    # ------------------------------------------------------------------------

    #: Display info on loading
    show_info_on_loading = Bool(True,
                                help='Display info on loading?'
                                ).tag(config=True)

    #: CSV data delimiter
    csv_delimiter = Unicode(';', help='CSV data delimiter').tag(config=True)

    #: Default DATA directory
    data = Unicode(help="Default data directory").tag(config=True)

    @default('data')
    def _get_data(self):
        return self._scpdata.data


class SpectroChemPy(Application):
    """
    This class SpectroChemPy is the main class, containing most of the setup,
    configuration and more.

    """
    from spectrochempy.utils import docstrings


    from spectrochempy.projects.projectpreferences import ProjectPreferences
    from spectrochempy.plotters.plotterpreferences import PlotterPreferences
    from spectrochempy.readers.readerpreferences import ReaderPreferences
    from spectrochempy.writers.writerpreferences import WriterPreferences
    from spectrochempy.processors.processorpreferences import ProcessorPreferences

    # applications attributes
    # ------------------------------------------------------------------------
    running = Bool(False)

    test = Bool(False)

    name = Unicode('SpectroChemPy')

    description = Unicode

    @default('description')
    def _get_description(self):
        desc = """Welcome to <strong>SpectroChemPy</strong> Application<br>
<br>
<p>
<strong>SpectroChemPy</strong> is a framework for processing, analysing and 
modelling 
<strong>Spectro</>scopic data for <strong>Chem</strong>istry with <strong>Py</strong>thon. It is 
is a cross platform software, running on Linux, Windows or OS X.
<br>
<br>
<strong>version:</strong> {version}
<br>
<strong>Authors:</strong> {authors}
<br>
<strong>License:</strong> {license}
<br>
<div class='warning'> SpectroChemPy is still experimental and under active 
development.
    Its current design is subject to major changes, reorganizations, bugs
    and crashes!!!. Please report any issues to the 
    <a url='https://bitbucket.org/spectrocat/spectrochempy'>Issue Tracker
    <a>
</div>
<br>
<br>
When using <strong>SpectroChemPy</strong> for your own work, you are kindly 
requested 
to cite it this way:
<pre>
 Arnaud Travert & Christian Fernandez,
 SpectroChemPy, a framework for processing, analysing and modelling of 
 Spectroscopic data for Chemistry with Python
 https://bitbucket.org/spectrocat/spectrochempy, (version {version})
 Laboratoire Catalyse and Spectrochemistry, ENSICAEN/University of
 Caen/CNRS, 2017
</pre>
</p>

""".format(version=__release__, authors=__author__, license=__license__)

        return desc


    # configuration parameters
    # ------------------------------------------------------------------------

    reset_config = Bool(False,
                        help='Should we restaure a default configuration?'
                        ).tag(config=True)

    config_file_name = Unicode(None,
                               help="Configuration file name"
                               ).tag(config=True)

    @default('config_file_name')
    def _get_config_file_name_default(self):
        return self.name.lower() + '.cfg.py'


    config_dir = Unicode(None,
                         help="Set the configuration directory location"
                         ).tag(config=True)

    @default('config_dir')
    def _get_config_dir_default(self):
        return self._get_config_dir()

    debug = Bool(False,
                 help='Set DEBUG mode, with full outputs'
                 ).tag(config=True)

    quiet = Bool(False,
                 help='Set Quiet mode, with minimal outputs'
                 ).tag(config=True)

    startup_project = Unicode('', help='Project to load at startup').tag(
        config=True)


    do_not_block = Bool(False,
                        help="Make the plots BUT do not stop (for tests)"
                        ).tag(config=True)

    aliases = Dict(
        dict(test='SpectroChemPy.test',
             p='SpectroChemPy.startup_project',
             log_level='SpectroChemPy.log_level'))

    flags = Dict(dict(
        debug=(
            {'SpectroChemPy': {'log_level': 10}},
            "Set loglevel to DEBUG")
    ))

    classes = List([GeneralPreferences,
                    ProjectPreferences,
                    PlotterPreferences,
                    ])

    # ------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------

    def __init__(self, *args, **kwargs):
        super(SpectroChemPy, self).__init__(*args, **kwargs)
        if kwargs.get('debug', False):
            self.log_level = logging.DEBUG

        self.initialize()

    # ------------------------------------------------------------------------
    # Initialisation of the application
    # ------------------------------------------------------------------------

    @catch_config_error
    def initialize(self, argv=None):
        """
        Initialisation function for the API applications

        Parameters
        ----------
        argv :  List, [optional].
            List of configuration parameters.

        """

        # parse the argv
        # --------------------------------------------------------------------

        # if we are running this under ipython and jupyter notebooks
        # deactivate potential command line arguments
        # (such that those from jupyter which cause problems here)

        self.log.debug('initialization of SpectroChemPy')

        _do_parse = True
        for arg in ['egg_info', '--egg-base',
                    'pip-egg-info', 'develop', '-f', '-x', '-c']:
            if arg in sys.argv:
                _do_parse = False

        if _do_parse:
            self.parse_command_line(sys.argv)

        # Get preferences from the config file
        # ---------------------------------------------------------------------

        if self.config_file_name:
            config_file = os.path.join(self.config_dir, self.config_file_name)
            self.load_config_file(config_file)

        # add other preferecnes
        # ---------------------------------------------------------------------

        self._init_general_preferences()
        self._init_plotter_preferences()
        self._init_project_preferences()

        # Test, Sphinx,  ...  detection
        # ---------------------------------------------------------------------

        for caller in ['builddocs.py', '-c']:
            # `-c` happen if the pytest is executed in parallel mode
            # using the plugin pytest-xdist

            if caller in sys.argv[0]:
                # this is necessary to build doc
                # with sphinx-gallery and doctests
                plt.ioff()
                self.do_not_block = True
                break

        for caller in ['pytest', 'py.test']:

            if caller in sys.argv[0]:
                # let's set do_not_block flag to true only if we are running
                #  the whole suite of tests
                if len(sys.argv)>1 and sys.argv[1].endswith("tests"):
                    plt.ioff()
                    self.do_not_block = True

        # case we have passed -test arguments to a script
        if len(sys.argv) > 1 and "-test" in sys.argv[1]:
            plt.ioff()
            self.do_not_block = True


        # we catch warnings and error for a ligther display to the end-user.
        # except if we are in debugging mode

        # warning handler
        # --------------------------------------------------------------------
        def send_warnings_to_log(message, category, filename,
                                 lineno,
                                 *args):
            self.log.warning(
                '%s:  %s' %
                (category.__name__, message))
            return

        warnings.showwarning = send_warnings_to_log

        # exception handler
        # --------------------------------------------------------------------
        ip = get_ipython()
        if ip is not None:

            def _custom_exc(shell, etype, evalue, tb,
                            tb_offset=None):
                if self.log_level == logging.DEBUG:
                    shell.showtraceback((etype, evalue, tb),
                                        tb_offset=tb_offset)
                else:
                    self.log.error(
                        "%s: %s" % (etype.__name__, evalue))

            ip.set_custom_exc((Exception,), _custom_exc)

        # load our custom magic extensions
        # --------------------------------------------------------------------
        if ip is not None:
            ip.register_magics(SpectroChemPyMagics)

        # Possibly write the default config file
        # --------------------------------------------------------------------
        self._make_default_config_file()

    # ------------------------------------------------------------------------
    # start the application
    # ------------------------------------------------------------------------

    @docstrings.get_sectionsf('SpectroChemPy.start')
    @docstrings.dedent
    def start(self, **kwargs):
        """
        Start the |scp| API or only make a plot if an `output` filename is
        given.

        Parameters
        ----------
        debug : `bool`
            Set application in debugging mode (log debug message
            are displayed in the standart output console).
        quiet : `bool`
            Set the application in minimal messaging mode. Only errors are
            displayed (bu no warnings). If both bebug and quiet are set
            (which is contradictory) debug has the priority.
        reset_config : `bool`
            Reset the configuration file to default values.

        Examples
        --------
        >>> app = SpectroChemPy()
        >>> app.initialize()
        >>> app.start(
        ...    reset_config=True,   # for restoring default configuration
        ...    debug=True,          # debugging logs
        ...    ) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        SpectroChemPy's API - v.0.1...
        True

        """

        try:

            if self.running:
                self.log.debug('API already started. Nothing done!')
                return

            for key in list(kwargs.keys()):
                if hasattr(self, key):
                    setattr(self, key, kwargs[key])

            self.log_format = '%(highlevel)s %(message)s'

            if self.quiet:
                self.log_level = logging.ERROR

            if self.debug:
                self.log_level = logging.DEBUG
                self.log_format = '[%(name)s %(asctime)s]%(highlevel)s %(message)s'

            info_string = "SpectroChemPy's API - v.{}\n" \
                          "© Copyright {}".format(__version__, __copyright__)

            # print(self.general_preferences.show_info_on_loading)
            if self.general_preferences.show_info_on_loading:
                print(info_string)

            self.log.debug(
                "The application was launched with ARGV : %s" % str(
                    sys.argv))

            self.running = True

            self.log.debug('MPL backend: {}'.format(mpl.get_backend()))

            return True

        except:

            return False

    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------

    # ........................................................................
    def _init_general_preferences(self):

        self.general_preferences = GeneralPreferences(config=self.config)

    # ........................................................................
    def _init_project_preferences(self):

        from spectrochempy.projects.projectpreferences import ProjectPreferences

        self.project_preferences = ProjectPreferences(config=self.config)

    # ........................................................................
    def _init_plotter_preferences(self):

        from spectrochempy.plotters.plotterpreferences import PlotterPreferences
        from spectrochempy.utils import install_styles

        # Pass config to other classes for them to inherit the config.
        self.plotter_preferences = PlotterPreferences(config=self.config)

        # also install style to be sure everything is set
        install_styles()


    # ........................................................................
    def _make_default_config_file(self):
        """auto generate default config file."""

        fname = config_file = os.path.join(self.config_dir,
                                           self.config_file_name)

        if not os.path.exists(fname) or self.reset_config:
            s = self.generate_config_file()
            self.log.warning("Generating default config file: %r" % fname)
            with open(fname, 'w') as f:
                f.write(s)

    # ........................................................................
    def _find_or_create_spectrochempy_dir(self, directory) :

        directory = os.path.join(os.path.expanduser('~'),
                                 '.spectrochempy', directory)

        if not os.path.exists(directory) :
            os.makedirs(directory, exist_ok=True)
        elif not os.path.isdir(directory) :
            msg = 'Intended SpectroChemPy directory `{0}` is ' \
                  'actually a file.'
            raise IOError(msg.format(directory))

        return os.path.abspath(directory)

    # ........................................................................
    def _get_config_dir(self, create=True) :
        """
        Determines the SpectroChemPy configuration directory name and
        creates the directory if it doesn't exist.

        This directory is typically ``$HOME/.spectrochempy/config``,
        but if the
        SCP_CONFIG_HOME environment variable is set and the
        ``$SCP_CONFIG_HOME`` directory exists, it will be that
        directory.

        If neither exists, the former will be created.

        Returns
        -------
        config_dir : str
            The absolute path to the configuration directory.

        """

        # first look for SCP_CONFIG_HOME
        scp = os.environ.get('SCP_CONFIG_HOME')

        if scp is not None and os.path.exists(scp) :
            return os.path.abspath(scp)

        return os.path.abspath(
            self._find_or_create_spectrochempy_dir('config'))

    # ------------------------------------------------------------------------
    # Events from Application
    # ------------------------------------------------------------------------

    @observe('log_level')
    def _log_level_changed(self, change):

        self.log_format = '%(highlevel)s %(message)s'
        if change.new == logging.DEBUG:
            self.log_format = '[%(name)s %(asctime)s]%(highlevel)s %(message)s'
        self.log.level = self.log_level
        for handler in self.log.handlers:
            handler.level = self.log_level
        self.log.debug("changed default log_level to {}".format(
                                             logging.getLevelName(change.new)))

#: Main application object that should not be called directly by a end user.
#: It is advisable to use the main `api` import to access all public methods of
#: this object.
app = SpectroChemPy()


# TODO: look at the subcommands capabilities of traitlets

if __name__ == "__main__":

    print('start application')
    pass

