import logging

logger = logging.getLogger(__name__)
logger.propagate = False
logger.info("This won't be shown if no handler is attached")
