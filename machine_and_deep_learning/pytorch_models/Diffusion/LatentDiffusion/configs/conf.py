import logging
from sys import stdout

# general configs
LOGD = logging.debug
LOGI = logging.info
LOGW = logging.warning
LOGE = logging.error
DEBUG_LEVEL = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}

# by default logs are sent to stdout
LOGFILES = stdout

# default resources path
RES_PATH = 'logs'