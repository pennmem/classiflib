Changes
=======

Version 1.4.0
-------------

**2023-03-29**

* Added support for python 3.11
* Removed support for all other python versions
* Added automatic testing and deployment to github with travis


Version 1.3.0
-------------

**2018-04-20**

* Rearranged ``OdinEmbeddedClassifierContainer`` once again to have channels as
  a simple list and moved weights data from channels to classifiers (#14)


Version 1.2.0
-------------

**2018-04-02**

* Made ``OdinEmbeddedClassifierContainer`` require a list of lists for channel
  specifications to support up to 2 classifiers (#12)


Version 1.1.0
-------------

**2018-03-09**

* Added a container for Odin embedded mode classifier data (#10)


Version 1.0.0
-------------

**2018-01-17**

* Implemented ``__eq__`` method for ``ClassifierContainer`` for quick
  comparisons (#4)
* Ensured that the ``ZipSerializer`` never pickles numpy arrays (#7)

Version 0.1.2
-------------

**2017-10-27**

* Bugfix release to handle exceptions when trying to get git revisions

Version 0.1.1
-------------

**2017-10-10**

* Use 0-based indexing in pairs dtype

Version 0.1.0
-------------

**2017-10-10**

Initial release.
