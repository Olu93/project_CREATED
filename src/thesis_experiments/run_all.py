import multiprocessing
import subprocess
import tensorflow as tf

# https://stackoverflow.com/a/62871460/4162265
# https://stackoverflow.com/a/62871460/4162265
from subprocess import call
from subprocess import Popen

scripts_1 = [
    "run_experiment_evolutionary_configs",
    "run_experiment_evolutionary_iterations",
    "run_experiment_evolutionary_params",
    "run_experiment_evolutionary_sidequest_configs",
    "run_experiment_evolutionary_sidequest_factuals",
    "run_experiment_evolutionary_sidequest_uniforms",
]


def work(cmd):
    return subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=count)
    with tf.device('/cpu:0'):
        pool.map(work, [f'python -m thesis_experiments.{scr}' for scr in scripts_1])