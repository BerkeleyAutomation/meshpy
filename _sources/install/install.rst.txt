Installation Instructions
=========================

Dependencies
~~~~~~~~~~~~
The `meshpy` module depends on the Berkeley AutoLab's `autolab_core`_ and `perception`_ modules,
which can be installed by following instructions in their respective
repositories.

.. _autolab_core: https://github.com/BerkeleyAutomation/autolab_core
.. _perception: https://github.com/BerkeleyAutomation/perception

Rendering using `meshpy` also depends on `OSMesa`_ and `Boost.NumPy`_, and
compiling the renderer depends on `CMake`_.

.. _OSMesa: http://www.mesa3d.org/osmesa.html
.. _Boost.NumPy: https://github.com/ndarray/Boost.NumPy
.. _CMake: https://cmake.org/

Any other dependencies will be installed automatically when `meshpy` is
installed with `pip`.

Cloning the Repository
~~~~~~~~~~~~~~~~~~~~~~
You can clone or download our source code from `Github`_. ::

    $ git clone git@github.com:BerkeleyAutomation/meshpy.git

.. _Github: https://github.com/BerkeleyAutomation/meshpy

Installation
~~~~~~~~~~~~
Install `OSMesa`_ by running: ::

    $ sudo apt-get install libosmesa6-dev   

Install `Boost.NumPy`_ by cloning the latest stable repo: ::

    $ git clone https://github.com/ndarray/Boost.NumPy.git

and following `Boost-Numpy's installation instructions`_.

.. _OSMesa: http://www.mesa3d.org/osmesa.html
.. _Boost.NumPy: https://github.com/ndarray/Boost.NumPy
.. _Boost-Numpy's installation instructions: https://github.com/ndarray/Boost.NumPy

Then to install `meshpy` in your current Python environment, simply
change directories into the `meshpy` repository and run ::

    $ pip install -e .

Alternatively, you can run ::

    $ pip install /path/to/meshpy

to install `meshpy` from anywhere.

To visualize meshes, we highly recommend also installing
the Berkeley AutoLab's `visualization`_ module, which uses `mayavi`_.
This can be installed by cloning the repo: ::

    $ git clone git@github.com:BerkeleyAutomation/visualization.git

and following `installation instructions`_.

.. _visualization: https://github.com/BerkeleyAutomation/visualization
.. _mayavi: http://docs.enthought.com/mayavi/mayavi/
.. _installation instructions: https://BerkeleyAutomation.github.io/visualization

Testing
~~~~~~~
To test your installation, run ::

    $ python setup.py test

We highly recommend testing before using the module.

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~
Building `meshpy`'s documentation requires a few extra dependencies --
specifically, `sphinx`_ and a few plugins.

.. _sphinx: http://www.sphinx-doc.org/en/1.4.8/

To install the dependencies required, simply run ::

    $ pip install -r docs_requirements.txt

Then, go to the `docs` directory and run ``make`` with the appropriate target.
For example, ::

    $ cd docs/
    $ make html

will generate a set of web pages. Any documentation files
generated in this manner can be found in `docs/build`.

Deploying Documentation
~~~~~~~~~~~~~~~~~~~~~~~
To deploy documentation to the Github Pages site for the repository,
simply push any changes to the documentation source to master
and then run ::

    $ . gh_deploy.sh

from the `docs` folder. This script will automatically checkout the
``gh-pages`` branch, build the documentation from source, and push it
to Github.
