import logging
from setup import SHOT_PATH
from logging.handlers import RotatingFileHandler
from logging import Formatter

logger = logging.getLogger('RotatingFileHandler')
logger.setLevel(logging.DEBUG)

handler = RotatingFileHandler(f'{SHOT_PATH}Log/log_filename.log', maxBytes=20000, backupCount=10)
formatter = Formatter('%(asctime)s - %(levelname)s - [in %(pathname)s:%(lineno)d] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# logger.debug('This is a debug log message.')
# logger.info('This is a info log message.')
# logger.warning('This is a warning log message.')
# logger.error('This is a error log message.')
# logger.critical('This is a critical log message.')