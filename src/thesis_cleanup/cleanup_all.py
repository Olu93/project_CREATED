import multiprocessing
import subprocess
import tensorflow as tf

# https://stackoverflow.com/a/62871460/4162265
from subprocess import call
from subprocess import Popen

scripts_1 = [
    "thesis_cleanup.cleanup_evo_iterations",
    "thesis_cleanup.cleanup_evo_params",
    "thesis_cleanup.cleanup_evo_configs",
    "thesis_cleanup.cleanup_evo_factuals",
]


def work(cmd):
    return subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=count)
    for script in scripts_1:
        with tf.device('/cpu:0'):

            p = Popen(
                ['conda activate ds && python -m', script],
                stdout=open(f'report_{script}.log', 'w'),
                stderr=open(f'error_{script}.log', 'w'),
                start_new_session=True,
            )
            # print(pool.map(work, ['ls'] * count), shell=True)