#!/bin/bash
# Split download file list into multiple files.  
# Arguments:
#   FILE:   file to split
#   NUM:    maximum number of lines in each file. Must be even number.
#   PREFIX: prefix to give each file.
#   OUTDIR: output directory to put files.


FILE=$1
NUM=$2
PREFIX=$3
OUTDIR=$4

split  -l $NUM $FILE $PREFIX 
mkdir -p $OUTDIR
mv $PREFIX* $OUTDIR/