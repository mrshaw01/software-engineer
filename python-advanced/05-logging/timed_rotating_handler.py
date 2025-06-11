import logging
from logging.handlers import TimedRotatingFileHandler
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = TimedRotatingFileHandler("timed_test.log", when="m", interval=1, backupCount=5)
logger.addHandler(handler)

for i in range(6):
    logger.info("Hello, world!")
    time.sleep(1)
