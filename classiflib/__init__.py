from collections import namedtuple
import logging

__version__ = '1.1.dev0'
version_info = namedtuple("VersionInfo", "major,minor,patch")(*__version__.split('.'))

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from .defaults import FRDefaults
from .container import ClassifierContainer
