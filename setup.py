import os
import setuptools
from setuptools.command.test import test as TestCommand
import codecs
import re
import sys

here = os.path.abspath(os.path.dirname(__file__))
def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(os.path.join(here, *parts), 'r').read()
def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
            version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['--strict', '--verbose', '--tb=long', 'tests']
        self.test_suite = True
    def run_tests(self):
            import pytest
            errno = pytest.main(self.test_args)
            sys.exit(errno)

long_description = read('README.rst')

setuptools.setup(
    name='top',
    version= '0.0.1', #find_version('top', '__init__.py'),
    packages=setuptools.find_packages(),
    author='Eder Santana',
    author_email='edercsjr+git@gmail.com',
    tests_require=['pytest'],
    description='top: Theano OPtimization module.',
    long_description=long_description,
    license='The MIT License',
    url='http://github.com/EderSantana/top/',
    install_requires=['numpy',
                      'scipy',
                      'theano'],
    cmdclass={'test': PyTest},
    extras_require={
        'testing': ['pytest'],
    },
    classifiers=['Development Status :: 1 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: MIT License',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering'],
)
