import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
)
# logging.basicConfig(filename="app.log") → log to file instead

logging.debug("Debug message")
