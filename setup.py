# This file is if we want to be able to pip install it & upload the package to pypi

from setuptools import setup


setup(name='ClusterAnalysis',
      version='0.1.0',
      description='Python package for analyzing clusters in molecular dynamics trajectories.',
      author='Jingyang Wang',
      author_email='jywang72@stanford.edu',
      url='https://github.com/exenGT/ClusterAnalysis',
      packages=['cluster_analysis'],
      license='MIT',
      python_requires='>=3.6',
      )
