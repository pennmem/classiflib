#!/usr/bin/env python

from __future__ import print_function

import os
import shutil
from subprocess import check_call

try:
    shutil.rmtree('build')
    os.mkdir('build')
except OSError:
    pass

build_cmd = "conda build -c pennmem conda.recipe --output-folder build/"
print(build_cmd)
check_call(build_cmd, shell=True)
