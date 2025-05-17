import logging

from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()

logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

logger.info("App started", extra={"user": "shawnguyen", "env": "prod"})
