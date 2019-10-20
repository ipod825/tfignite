import os
import sys
from pathlib import Path


def makedirp(*names):
    for name in names:
        if '~' in name:
            name = name.replace('~', Path.home())
        if not os.path.isdir(name):
            os.makedirs(name)


def stop_if_no(message, double_check=False):
    ans = input(message + ' (y/n)')
    if ans != "y":
        sys.exit(0)
    if double_check:
        ans = input("Are you sure? (y/n)")
        if ans != "y":
            sys.exit(0)
