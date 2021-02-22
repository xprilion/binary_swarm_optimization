#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
echo "* Hello world from compute server `hostname`!"
echo "* The current directory is ${PWD}."
echo "* Compute server's CPU model and number of logical CPUs:"
lscpu | grep 'Model name\\|^CPU(s)'
echo "* Python available to us:"
which python
python --version
python main.py
echo "*Bye"
