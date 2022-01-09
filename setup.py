from setuptools import setup, find_packages, Extension
import pathlib
# Follows https://www.youtube.com/watch?v=zhpI6Yhz9_4
# Follows https://packaging.python.org/en/latest/tutorials/packaging-projects/

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python :: 3.7',
]
CURR_DIR = pathlib.Path('.')

print(__package__)
print(f"Found packages: {find_packages(where='src')}")
setup(
    name='thesis_data_readers',
    version='0.0.1',
    author="Olusanmi Hundogan",
    author_email="o.hundogan@gmail.com",
    description='A reader package for specific logs within process mining',
    long_description=open(CURR_DIR / 'README.md').read() + '\n\n' + open(CURR_DIR / 'CHANGELOG.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/Olu93/thesis_data_readers',
    project_urls={
        "Bug Tracker": "https://github.com/Olu93/thesis_data_readers/issues",
    },
    classifiers=classifiers,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=open(CURR_DIR / 'requirements.txt').read().split('\n'),
    python_requires='>=3.6',
    keywords="reader, xes, log, BPIC",
    package_data={'': ['data/preprocessed/*.csv']}
)
