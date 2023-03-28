#!/usr/bin/env python

import glob
import shlex
from subprocess import check_call

for platform in ['linux-64', 'linux-aarch64', 'osx-64', 'osx-arm64', 'win-64']:
    files = glob.glob('build/{}/*.tar.bz2'.format(platform))
    cmds = ['anaconda upload -u pennmem {}'.format(f) for f in files]
    for cmd in cmds:
        print(cmd)
        check_call(shlex.split(cmd))
