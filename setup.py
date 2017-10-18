#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from classiflib import __version__

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.rst') as history_file:
    history = history_file.read()

# TODO: put package requirements here
requirements = []

# TODO: put setup requirements here
setup_requirements = []

setup(
    name='classiflib',
    version=__version__,
    description="Memory state classifier utilities",
    long_description=readme + '\n\n' + history,
    author="Penn Computational Memory Lab",
    url='https://github.com/pennmem/classiflib',
    packages=find_packages(include=['classiflib']),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='classiflib',
    setup_requires=setup_requirements,
)
