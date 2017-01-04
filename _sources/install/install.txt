Installation Instructions
=========================

Dependencies
~~~~~~~~~~~~
The `meshpy` module depends on the Berkeley AutoLab's `core`_ and `perception`_ modules,
which can be installed using `pip install` on the source repo.

.. _core: https://github.com/mmatl/core
.. _perception: https://github.com/mmatl/perception

Rendering using `meshpy` also depends on `OSMesa`_ and `Boost.NumPy`_.

.. _OSMesa: http://www.mesa3d.org/osmesa.html
.. _Boost.NumPy: https://github.com/ndarray/Boost.NumPy

Any other dependencies will be installed automatically when `meshpy` is
installed with `pip`.

Cloning the Repository
~~~~~~~~~~~~~~~~~~~~~~
You can clone or download our source code from `Github`_. ::

    $ git clone git@github.com:mmatl/meshpy.git

.. _Github: https://github.com/mmatl/meshpy

Installation
~~~~~~~~~~~~
Install `OSMesa`_ by running:

    $ sudo apt-get install libosmesa6-dev   

Install `Boost.NumPy`_ by cloning the latest stable repo:

    $ git clone https://github.com/ndarray/Boost.NumPy.git

and following the `installation instructions`_.

.. _OSMesa: http://www.mesa3d.org/osmesa.html
.. _Boost.NumPy: https://github.com/ndarray/Boost.NumPy
.. _installation instructions: https://github.com/ndarray/Boost.NumPy

Then to install `meshpy` in your current Python environment, simply
change directories into the `meshpy` repository and run ::

    $ pip install -e .

or ::

    $ pip install -r requirements.txt

Alternatively, you can run ::

    $ pip install /path/to/meshpy

to install `meshpy` from anywhere.

To visualize meshes, we highly recommend also installing the Berkeley AutoLab's `visualization`_ module, which uses `mayavi`_.
This can be installed by cloning the repo:

    $ git clone git@github.com:jeffmahler/visualization.git

and following `installation instructions`_.

.. _visualization: https://github.com/jeffmahler/visualization
.. _mayavi: http://docs.enthought.com/mayavi/mayavi/
.. _installation instructions: https://jeffmahler.github.io/visualization

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

Then, go to the `docs` directory and run `make` with the appropriate target.
For example, ::

    $ cd docs/
    $ make html

will generate a set of web pages. Any documentation files
generated in this manner can be found in `docs/build`.

