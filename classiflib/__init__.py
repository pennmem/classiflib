from collections import namedtuple
import logging

__version__ = '0.1.2'
version_info = namedtuple("VersionInfo", "major,minor,patch")(*__version__.split('.'))

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from .defaults import FRDefaults
from .container import ClassifierContainer
