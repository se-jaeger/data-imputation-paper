# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'data-imputation-paper'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound


import sys
import logging

# setup logging
logger = logging.getLogger()

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s - %(filename)-12s:%(lineno)-4s - %(levelname)-7s - %(funcName)-15s]: %(message)s")

info_handler = logging.StreamHandler(sys.stdout)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

error_handler = logging.StreamHandler(sys.stderr)
error_handler.setLevel(logging.WARNING)
error_handler.setFormatter(formatter)

logger.addHandler(error_handler)
logger.addHandler(info_handler)
