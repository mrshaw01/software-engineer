import logging.config

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("simpleExample")

logger.debug("debug message")
logger.info("info message")
