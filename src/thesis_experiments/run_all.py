import multiprocessing
import subprocess

def work(cmd):
    return subprocess.call(cmd, shell=False)

if __name__ == '__main__':
    count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=count)
    print(pool.map(work, ['ls'] * count))