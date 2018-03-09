classiflib
==========

Serialization format
--------------------

.. image:: serialization_spec.svg
    :target: ../serialization_spec.svg

Workflow
--------

Saving a classifier
^^^^^^^^^^^^^^^^^^^

* Train a ``LogisticRegression`` classifier from ``sklearn.linear_model``
* Create a ``classiflib.ClassifierContainer`` to store the data required to
  recreate a trained classifier (this includes some extra metadata)
* Save the container as a zip file (see API docs for other formats)

Loading a classifier
^^^^^^^^^^^^^^^^^^^^

* Load with the :meth:`classiflib.ClassifierContainer.load` class method

This automatically sets the weights and intercept, so no additional steps are
required if the classifier does not need to be retrained on newly excluded
pairs.

Structured data types
---------------------

Some data are stored with Numpy record arrays. The data types for these arrays
are defined in :mod:`classiflib.dtypes`:

.. autodata:: classiflib.dtypes.pairs
.. autodata:: classiflib.dtypes.weights
.. autodata:: classiflib.dtypes.timing_window

.. note:: When adding pairs to a :class:`ClassifierContainer`, an additional
          ``id`` column is automatically added. This is used to reference the
          bipolar pairs in the ``weights`` array.

Odin embedded mode
------------------

Odin embedded mode data types use the :mod:`traitschema` package for easier
serializability.

.. autoclass:: classiflib.dtypes.OdinEmbeddedMeta
    :members:

.. autoclass:: classiflib.dtypes.OdinEmbeddedClassifier
    :members:

.. autoclass:: classiflib.dtypes.OdinEmbeddedChannel
    :members:

Rather than using :class:`classiflib.ClassifierContainer`, use
:class:`classiflib.OdinEmbeddedClassifierContainer`:

.. autoclass:: classiflib.container.OdinEmbeddedClassifierContainer
    :members:


Utilities
---------

.. autofunction:: classiflib.util.convert_pairs_json

Experiment defaults
-------------------

For convenience, timing windows for some experiments are perdefined in
:mod:`classiflib.defaults`.

.. autoclass:: classiflib.defaults.FRDefaults
    :members:
    :undoc-members:

API reference
-------------

.. automodule:: classiflib.container
    :members:
