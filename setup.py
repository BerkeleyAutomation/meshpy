"""
Setup of meshpy python codebase
Author: Jeff Mahler
"""
from setuptools import setup, Extension

requirements = [
    'numpy',
    'perception',
    'core',
    'scipy',
    'sklearn',
    'Pillow',
    'nearpy'
]

meshrender = Extension('meshrender',
                       include_dirs = ['/usr/include',
                                        '${PYTHONPATH}'],
                       libraries = ['boost_python',
                                    'python2.7',
                                    'boost_numpy',
                                    'glut',
                                    'OSMesa'],
                       sources = ['meshpy/meshrender.cpp'])

setup(name='meshpy',
      version='0.1.dev0',
      description='MeshPy project code',
      author='Matt Matl',
      author_email='mmatl@berkeley.edu',
      package_dir = {'': '.'},
      packages=['meshpy'],
      ext_modules = [meshrender],
      install_requires=requirements,
      test_suite='test'
     )
