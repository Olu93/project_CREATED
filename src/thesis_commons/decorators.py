import datetime
import time
from functools import wraps


# https://stackoverflow.com/a/36944992/4162265
def collect_time_stat(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        duration = time.time() - start_time
        self.time_stats[func.__name__] = duration
        return result
    return inner