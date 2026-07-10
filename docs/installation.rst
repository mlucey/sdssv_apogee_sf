Installation
============

.. code-block:: bash

   pip install sdssv-apogee-sf

Or from source::

   git clone https://github.com/YOUR_USERNAME/sdssv_apogee_sf
   cd sdssv_apogee_sf
   pip install -e .

Dependencies
------------

- ``numpy``
- ``astropy``
- ``healpy``
- ``requests``
- ``tqdm``
- ``sdss-semaphore`` (for filtering ASPCAP catalogues by programme)

Optional (for building the docs)::

   pip install sphinx sphinx-rtd-theme
