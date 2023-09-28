# -*- coding: utf-8 -*-
# file: setup.py
# date: 2023-09-28


import os
import sys
import re
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
from typing import List, Tuple, Dict


CURR_DIR: str = os.path.abspath(os.path.dirname(__file__))


def get_install_deps() -> Tuple[List[str], List[str] ]:
    dependency_links: List[str] = []
    install_requires: List[str] = []
    requirements: List[str] = open(
        os.path.join(CURR_DIR, "requirements.txt"), "r")\
                .read().strip("\n").strip(" ").split("\n")

    for dep in requirements:
        if "git@" in dep or "://" in dep:
            dependency_links.append(dep)
        else:
            install_requires.append(dep)
    return (install_requires, dependency_links)


if __name__ == "__main__":
    get_install_deps()
    setup(
        name="nnfeakit",
        version="0.0.0",
        author="innerNULL",
        author_email="",
        description="Neural Network Feaature Kit",
        url="https://github.com/innerNULL/nnfeakit",
        python_requires='>=3.8, <=3.11',
        install_requires=get_install_deps()[0], 
        dependency_links=get_install_deps()[1],
        packages=find_packages(),
        zip_safe=False,
        extras_require={"test": ["pytest"]},
    )
