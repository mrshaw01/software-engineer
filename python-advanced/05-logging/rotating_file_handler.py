import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = RotatingFileHandler("app.log", maxBytes=2000, backupCount=5)
logger.addHandler(handler)

for _ in range(10000):
    logger.info("Hello, world!")
