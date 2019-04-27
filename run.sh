#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################
E_BADARGS=65
if [ $# -ne 1 ]
then
	echo "Usage: `basename $0` <input>"
	exit $E_BADARGS
fi

INPUT=$1
./src/pypy3.6-v7.1.1-linux64/bin/pypy3 ./src/local-search.py $INPUT
