"""
Setup of meshpy python codebase
Author: Jeff Mahler
"""
from setuptools import setup

setup(name='meshpy',
      version='0.1.dev0',
      description='MeshPy project code',
      author='Matt Matl',
      author_email='mmatl@berkeley.edu',
      package_dir = {'': '.'},
      packages=['meshpy'],
      test_suite='test'
     )
