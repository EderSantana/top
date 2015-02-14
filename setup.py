import os
import setuptools

setuptools.setup(
    name='top',
    version='0.1',
    packages=setuptools.find_packages(),
    author='Eder Santana',
    author_email='edercsjr+git@gmail.com',
    description='top: Theano OPtimization module.',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.rst')).read(),
    license='The MIT License',
    url='http://github.com/EderSantana/top/',
    install_requires=['numpy',
                      'scipy',
                      'theano],
    classifiers=['Development Status :: 1 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: MIT License',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering'],
)
