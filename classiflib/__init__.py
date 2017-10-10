from collections import namedtuple

__version__ = '0.1.1'
version_info = namedtuple("VersionInfo", "major,minor,patch")(*__version__.split('.'))

from .defaults import FRDefaults
from .container import ClassifierContainer
