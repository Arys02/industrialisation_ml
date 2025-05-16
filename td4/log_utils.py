import logging
import time
from functools import wraps


def log_duration(name=None, level=logging.INFO):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            label = name if name else func.__name__
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = round((time.time() - start_time) * 1000, 2)
            logging.log(level, f"'{label}' exécutée en {duration} ms")
            logging.log
            return result
        return wrapper
    return decorator
