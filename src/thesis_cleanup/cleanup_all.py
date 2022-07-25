import multiprocessing
import subprocess
import tensorflow as tf

# https://stackoverflow.com/a/62871460/4162265
from subprocess import call
from subprocess import Popen

scripts_1 = [
    "cleanup_evo_iterations",
    "cleanup_evo_params",
    "cleanup_evo_configs",
    "cleanup_evo_factuals",
]


def work(cmd):
    return subprocess.call(cmd, shell=False)


if __name__ == '__main__':
    count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=count)
    for script in scripts_1:
        with tf.device('/cpu:0'):

            p = Popen(
                ['nohup', 'python -m', script],
                stdout=open('null1', 'w'),
                stderr=open(f'{"error_{script}"}.log', 'a'),
                start_new_session=True,
            )
            # print(pool.map(work, ['ls'] * count), shell=True)