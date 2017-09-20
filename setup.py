#!/usr/bin/env python3

from setuptools import setup

setup(
    name="Nuc Analyze",
    version="0.1.5",
    description="A tool to collect stats about Hi-C structures",
    packages=['nuc_analyze'],
    entry_points={'console_scripts': ['nuc_analyze=nuc_analyze.main:cli']}
)
