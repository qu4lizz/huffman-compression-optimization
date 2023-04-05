#!/bin/bash

# get files from resources
inputFiles=resources/*
# get output directory
outputDir=out

for inputFile in $inputFiles
do
    echo "Running $inputFile"
    for i in {1..4}
    do
        echo "Run $i"
        ./run.sh $inputFile "$outputDir/out"
    done
    echo "Done running $inputFile"
    echo ''
done