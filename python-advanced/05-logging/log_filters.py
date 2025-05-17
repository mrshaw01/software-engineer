import logging


class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.addFilter(InfoFilter())
logger.addHandler(handler)

logger.info("This will appear")
logger.warning("This won't")
