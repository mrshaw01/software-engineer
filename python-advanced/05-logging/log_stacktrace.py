import logging
import traceback

try:
    a = [1, 2, 3]
    value = a[3]
except IndexError as e:
    logging.error("Exception occurred", exc_info=True)
    logging.error("Traceback:\n%s", traceback.format_exc())
