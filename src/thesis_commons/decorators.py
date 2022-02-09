import time
from functools import wraps
import datetime

# https://stackoverflow.com/a/36944992/4162265
def collect_time_stat(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        duration = time.time() - start_time
        self.time_stats[func.__name__] = str(datetime.timedelta(seconds=duration))
        return result
    return inner